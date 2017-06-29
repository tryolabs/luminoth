import collections
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim


from sonnet.python.modules.conv import Conv2D

from .anchor_target import AnchorTarget
from .rpn_proposal import RPNProposal
from .utils.generate_anchors import generate_anchors
from .utils.losses import smooth_l1_loss
from .utils.ops import spatial_softmax, spatial_reshape_layer
from .utils.vars import variable_summaries


class RPN(snt.AbstractModule):

    def __init__(self, num_anchors, num_channels=512, kernel_shape=[3, 3], name='rpn'):
        """RPN - Region Proposal Network

        This module works almost independently from the Faster RCNN module.
        It instantiates its own submodules and calculates its own loss,
        and can be used on its own

        """
        super(RPN, self).__init__(name=name)
        self._num_anchors = num_anchors
        self._num_channels = num_channels
        self._kernel_shape = kernel_shape

        # According to Faster RCNN paper we need to initialize layers with
        # "from a zero-mean Gaussian distribution with standard deviation 0.01"
        self._initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        # We could use normal relu without any problems.
        self._rpn_activation = tf.nn.relu6

        self._instantiate_layers()

    def _instantiate_layers(self):
        """Instantiates all convolutional modules used in the RPN."""
        with self._enter_variable_scope():
            self._rpn = Conv2D(
                output_channels=self._num_channels,
                kernel_shape=self._kernel_shape,
                initializers={'w': self._initializer}, name='conv'
            )

            self._rpn_cls = Conv2D(
                output_channels=self._num_anchors * 2, kernel_shape=[1, 1],
                initializers={'w': self._initializer}, padding='VALID',
                name='cls_conv'
            )

            # BBox prediction is 4 values * number of anchors.
            self._rpn_bbox = Conv2D(
                output_channels=self._num_anchors * 4, kernel_shape=[1, 1],
                initializers={'w': self._initializer}, padding='VALID',
                name='bbox_conv'
            )


            self._proposal = RPNProposal(self._num_anchors)
            self._anchor_target = AnchorTarget(self._num_anchors)

    def _build(self, pretrained_feature_map, gt_boxes, image_shape, all_anchors,
               is_training=True):
        # We start with a common conv layer applied to the feature map.
        rpn_feature = self._rpn_activation(self._rpn(pretrained_feature_map))

        # Then we apply separate conv layers for classification and regression.
        rpn_cls_score_original = self._rpn_cls(rpn_feature)
        rpn_bbox_pred_original = self._rpn_bbox(rpn_feature)
        # rpn_cls_score_original has shape (1, H, W, num_anchors * 2)
        # rpn_bbox_pred_original has shape (1, H, W, num_anchors * 4)
        # where H, W are height and width of the pretrained feature map.

        # Convert `rpn_cls_score` which has two scalars per anchor per location
        # to be able to apply `spatial_softmax`.
        rpn_cls_score = tf.reshape(rpn_cls_score_original, [-1, 2])
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score)

        # Flatten bounding box delta prediction for easy manipulation.
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred_original, [-1, 4])

        # We have to convert bbox deltas to usable bounding boxes and remove
        # redudant bbox using non maximum suppression.
        proposal_prediction = self._proposal(
            rpn_cls_prob, rpn_bbox_pred, all_anchors, image_shape)

        if is_training:
            # When training we use a separate module to calculate the target
            # values we want to output.
            rpn_cls_target, rpn_bbox_target, rpn_max_overlap = self._anchor_target(
                tf.shape(pretrained_feature_map), gt_boxes, image_shape, all_anchors)

        # TODO: Better way to log variable summaries.
        variable_summaries(self._rpn.w, 'rpn_conv_W', ['RPN'])
        variable_summaries(self._rpn_cls.w, 'rpn_cls_W', ['RPN'])
        variable_summaries(self._rpn_bbox.w, 'rpn_bbox_W', ['RPN'])

        variable_summaries(proposal_prediction['nms_proposals_scores'], 'rpn_scores', ['RPN'])
        variable_summaries(rpn_cls_prob, 'rpn_cls_prob', ['RPN'])

        # TODO: Remove unnecesary variables from prediction dictionary.
        prediction_dict = {
            'rpn_feature': rpn_feature,
            'rpn_cls_prob': rpn_cls_prob,
            'rpn_cls_score': rpn_cls_score,
            'rpn_cls_score_original': rpn_cls_score_original,
            'rpn_bbox_pred': rpn_bbox_pred,
            'rpn_bbox_pred_original': rpn_bbox_pred_original,
            'proposals': proposal_prediction['nms_blobs'],
            'scores': proposal_prediction['nms_proposals_scores'],
            'proposal_prediction': proposal_prediction,
        }

        if is_training:
            prediction_dict['rpn_cls_target'] = rpn_cls_target
            prediction_dict['rpn_bbox_target'] = rpn_bbox_target
            prediction_dict['rpn_max_overlap'] = rpn_max_overlap

        return prediction_dict

    def loss(self, prediction_dict):
        """
        Returns cost for Region Proposal Network based on:

        Args:
            rpn_cls_prob: Probability of for being an object for each anchor
                in the image. Shape -> (num_anchors, 2)
            rpn_cls_target: Ground truth labeling for each anchor. Should be
                1: for positive labels
                0: for negative labels
                -1: for labels we should ignore.
                Shape -> (num_anchors, 4)
            rpn_bbox_target: Bounding box output delta target for rpn.
                Shape -> (num_anchors, 4)
            rpn_bbox_pred: Bounding box output delta prediction for rpn.
                Shape -> (num_anchors, 4)
        Returns:
            Multiloss between cls probability and bbox target.
        """

        rpn_cls_prob = prediction_dict['rpn_cls_prob']
        rpn_cls_score = prediction_dict['rpn_cls_score']
        rpn_cls_target = prediction_dict['rpn_cls_target']

        rpn_bbox_target = prediction_dict['rpn_bbox_target']
        rpn_bbox_pred = prediction_dict['rpn_bbox_pred']

        # First, we need to calculate classification loss over `rpn_cls_prob`
        # and `rpn_cls_target`. Ignoring all anchors where `rpn_cls_target =
        # -1`.

        # For classification loss we use log loss of two classes. So we need to:
        # - filter `rpn_cls_prob` that are ignored. We need to reshape both labels and prob
        # - transform positive and negative `rpn_cls_target` to same shape as `rpn_cls_prob`.
        # - then we can use `tf.losses.log_loss` which returns a tensor.

        with self._enter_variable_scope():
            with tf.name_scope('RPNLoss'):
                # Flatten already flat Tensor for usage as boolean mask filter.
                rpn_cls_target = tf.cast(tf.reshape(
                    rpn_cls_target, [-1]), tf.int32, name='rpn_cls_target')
                # Transform to boolean tensor with True only when != -1 (else
                # == -1 -> False)
                labels_not_ignored = tf.not_equal(
                    rpn_cls_target, -1, name='labels_not_ignored')

                # Now we only have the labels we are going to compare with the
                # cls probability.
                labels = tf.boolean_mask(rpn_cls_target, labels_not_ignored)
                cls_prob = tf.boolean_mask(rpn_cls_prob, labels_not_ignored)
                cls_score = tf.boolean_mask(rpn_cls_score, labels_not_ignored)

                # We need to transform `labels` to `cls_prob` shape.
                # convert [1, 0] to [[0, 1], [1, 0]]
                cls_target = tf.one_hot(labels, depth=2)

                cross_entropy_per_anchor = tf.nn.softmax_cross_entropy_with_logits(
                    labels=cls_target, logits=cls_score
                )

                prediction_dict['cross_entropy_per_anchor'] = cross_entropy_per_anchor

                # Finally, we need to calculate the regression loss over
                # `rpn_bbox_target` and `rpn_bbox_pred`.
                # Since `rpn_bbox_target` is obtained from AnchorTargetLayer then we
                # just need to apply SmoothL1Loss.
                rpn_bbox_target = tf.reshape(rpn_bbox_target, [-1, 4])
                rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

                # We only care for positive labels (we ignore backgrounds since
                # we don't have any bounding box information for it).
                positive_labels = tf.equal(rpn_cls_target, 1)
                rpn_bbox_target = tf.boolean_mask(rpn_bbox_target, positive_labels)
                rpn_bbox_pred = tf.boolean_mask(rpn_bbox_pred, positive_labels)

                # We apply smooth l1 loss as described by the Fast R-CNN paper.
                reg_loss_per_anchor = smooth_l1_loss(rpn_bbox_pred, rpn_bbox_target)

                prediction_dict['reg_loss_per_anchor'] = reg_loss_per_anchor

                return {
                    'rpn_cls_loss': tf.reduce_mean(cross_entropy_per_anchor),
                    'rpn_reg_loss': tf.reduce_mean(reg_loss_per_anchor),
                }
