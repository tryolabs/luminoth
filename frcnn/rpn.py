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

        # TODO: Do we need the anchors? Can't we just use
        # len(self._anchor_scales) * len(self._anchor_ratios)
        self._num_anchors = num_anchors

        self._num_channels = num_channels
        self._kernel_shape = kernel_shape

        self._initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self._rpn_activation = tf.nn.relu

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

            # AnchorTarget and RPNProposal and RPN modules and should live in this scope
            # TODO: Anchor target only in training. Is there a problem if we
            # create it when not training?
            self._anchor_target = AnchorTarget(self._num_anchors)
            self._proposal = RPNProposal(self._num_anchors)

    def _build(self, pretrained, gt_boxes, image_shape, all_anchors, is_training=True):
        """
        TODO: We don't have BatchNorm yet.
        """
        rpn = self._rpn_activation(self._rpn(pretrained))
        rpn_cls_score = self._rpn_cls(rpn)
        rpn_cls_score_reshape = spatial_reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = spatial_softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = spatial_reshape_layer(
            rpn_cls_prob, self._num_anchors * 2)
        rpn_bbox_pred = self._rpn_bbox(rpn)

        rpn_labels, rpn_bbox = self._anchor_target(
            rpn_cls_prob_reshape, gt_boxes, image_shape, all_anchors)

        proposals, scores = self._proposal(
            rpn_cls_prob_reshape, rpn_bbox_pred, all_anchors, image_shape)

        variable_summaries(self._rpn.w, 'rpn_conv_W', ['RPN'])
        variable_summaries(self._rpn_cls.w, 'rpn_cls_W', ['RPN'])
        variable_summaries(self._rpn_bbox.w, 'rpn_bbox_W', ['RPN'])

        variable_summaries(scores, 'rpn_scores', ['RPN'])
        variable_summaries(rpn_cls_prob_reshape, 'rpn_cls_prob', ['RPN'])

        return {
            'rpn': rpn,
            'rpn_cls_prob': rpn_cls_prob,
            'rpn_cls_prob_reshape': rpn_cls_prob_reshape,
            'rpn_cls_score': rpn_cls_score,
            'rpn_cls_score_reshape': rpn_cls_score_reshape,
            'rpn_bbox_pred': rpn_bbox_pred,
            'rpn_cls_target': rpn_labels,
            'rpn_bbox_target': rpn_bbox,
            'proposals': proposals,
            'scores': scores,
        }

    def loss(self, prediction_dict):
        """
        Returns cost for Region Proposal Network based on:

        Args:
            rpn_cls_prob: Probability of for being an object for each anchor
                in the image. Shape -> (1, height, width, 2)
            rpn_cls_target: Ground truth labeling for each anchor. Should be
                1: for positive labels
                0: for negative labels
                -1: for labels we should ignore.
                Shape -> (1, height, width, 4)
            rpn_bbox_target: Bounding box output target for rpn.
            rpn_bbox_pred: Bounding box output prediction for rpn.

        Returns:
            Multiloss between cls probability and bbox target.
        """

        rpn_cls_prob = prediction_dict['rpn_cls_prob']
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
                # Flatten labels.
                rpn_cls_target = tf.cast(tf.reshape(
                    rpn_cls_target, [-1]), tf.int32, name='rpn_cls_target')
                # Transform to boolean tensor with True only when != -1 (else
                # == -1 -> False)
                labels_not_ignored = tf.not_equal(
                    rpn_cls_target, -1, name='labels_not_ignored')

                # Flatten rpn_cls_prob (only anchors, not completely).
                rpn_cls_prob = tf.reshape(
                    rpn_cls_prob, [-1, 2], name='rpn_cls_prob_flatten')

                # Now we only have the labels we are going to compare with the
                # cls probability. We need to remove the background.
                labels = tf.boolean_mask(
                    rpn_cls_target, labels_not_ignored, name='labels')
                cls_prob = tf.boolean_mask(rpn_cls_prob, labels_not_ignored)

                # We need to transform `labels` to `cls_prob` shape.
                cls_target = tf.one_hot(labels, depth=2)

                # TODO: In other implementations they use
                # `sparse_softmax_cross_entropy_with_logits` with
                # `reduce_mean`. Should we use that?
                log_loss = tf.losses.log_loss(cls_target, cls_prob)
                # TODO: For logs
                cls_loss = tf.identity(log_loss, name='log_loss')

                # Finally, we need to calculate the regression loss over `rpn_bbox_target`
                # and `rpn_bbox_pred`.
                # Since `rpn_bbox_target` is obtained from AnchorTargetLayer then we
                # just need to apply SmoothL1Loss.
                rpn_bbox_target = tf.reshape(rpn_bbox_target, [-1, 4])
                rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

                # We only care for positive labels
                positive_labels = tf.equal(rpn_cls_target, 1)
                rpn_bbox_target = tf.boolean_mask(
                    rpn_bbox_target, positive_labels)
                rpn_bbox_pred = tf.boolean_mask(rpn_bbox_pred, positive_labels)

                reg_loss = smooth_l1_loss(rpn_bbox_pred, rpn_bbox_target)

                return {
                    'rpn_cls_loss': cls_loss,
                    'rpn_reg_loss': reg_loss,
                }
