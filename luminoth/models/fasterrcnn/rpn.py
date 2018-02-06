"""
RPN - Region Proposal Network
"""

import sonnet as snt
import tensorflow as tf

from sonnet.python.modules.conv import Conv2D

from .rpn_target import RPNTarget
from .rpn_proposal import RPNProposal
from luminoth.utils.losses import smooth_l1_loss
from luminoth.utils.vars import (
    get_initializer, layer_summaries, variable_summaries,
    get_activation_function
)


class RPN(snt.AbstractModule):

    def __init__(self, num_anchors, config, debug=False, seed=None,
                 name='rpn'):
        """RPN - Region Proposal Network.

        Given an image (as feature map) and a fixed set of anchors, the RPN
        will learn weights to adjust those anchors so they better look like the
        ground truth objects, as well as scoring them by "objectness" (ie. how
        likely they are to be an object vs background).

        The final result will be a set of rectangular boxes ("proposals"),
        each associated with an objectness score.

        Note: this module can be used independently of Faster R-CNN.
        """
        super(RPN, self).__init__(name=name)
        self._num_anchors = num_anchors
        self._num_channels = config.num_channels
        self._kernel_shape = config.kernel_shape

        self._debug = debug
        self._seed = seed

        self._rpn_initializer = get_initializer(
            config.rpn_initializer, seed=seed
        )
        # According to Faster RCNN paper we need to initialize layers with
        # "from a zero-mean Gaussian distribution with standard deviation 0.01
        self._cls_initializer = get_initializer(
            config.cls_initializer, seed=seed
        )
        self._bbox_initializer = get_initializer(
            config.bbox_initializer, seed=seed
        )
        self._regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.l2_regularization_scale
        )

        self._l1_sigma = config.l1_sigma

        # We could use normal relu without any problems.
        self._rpn_activation = get_activation_function(
            config.activation_function
        )

        self._config = config

    def _instantiate_layers(self):
        """Instantiates all convolutional modules used in the RPN."""
        self._rpn = Conv2D(
            output_channels=self._num_channels,
            kernel_shape=self._kernel_shape,
            initializers={'w': self._rpn_initializer},
            regularizers={'w': self._regularizer},
            name='conv'
        )

        self._rpn_cls = Conv2D(
            output_channels=self._num_anchors * 2, kernel_shape=[1, 1],
            initializers={'w': self._cls_initializer},
            regularizers={'w': self._regularizer},
            padding='VALID', name='cls_conv'
        )

        # BBox prediction is 4 values * number of anchors.
        self._rpn_bbox = Conv2D(
            output_channels=self._num_anchors * 4, kernel_shape=[1, 1],
            initializers={'w': self._bbox_initializer},
            regularizers={'w': self._regularizer},
            padding='VALID', name='bbox_conv'
        )

    def _build(self, conv_feature_map, im_shape, all_anchors,
               gt_boxes=None, is_training=False):
        """Builds the RPN model subgraph.

        Args:
            conv_feature_map: A Tensor with the output of some pretrained
                network. Its dimensions should be
                `[1, feature_map_height, feature_map_width, depth]` where depth
                is 512 for the default layer in VGG and 1024 for the default
                layer in ResNet.
            im_shape: A Tensor with the shape of the original image.
            all_anchors: A Tensor with all the anchor bounding boxes. Its shape
                should be
                [feature_map_height * feature_map_width * total_anchors, 4]
            gt_boxes: A Tensor with the ground-truth boxes for the image.
                Its dimensions should be `[total_gt_boxes, 5]`, and it should
                consist of [x1, y1, x2, y2, label], being (x1, y1) -> top left
                point, and (x2, y2) -> bottom right point of the bounding box.

        Returns:
            prediction_dict: A dict with the following keys:
                proposals: A Tensor with a variable number of proposals for
                    objects on the image.
                scores: A Tensor with a "objectness" probability for each
                    proposal. The score should be the output of the softmax for
                    object.

                If training is True, then some more Tensors are added to the
                prediction dictionary to be used for calculating the loss.

                rpn_cls_prob: A Tensor with the probability of being
                    background and foreground for each anchor.
                rpn_cls_score: A Tensor with the cls score of being background
                    and foreground for each anchor (the input for the softmax).
                rpn_bbox_pred: A Tensor with the bounding box regression for
                    each anchor.
                rpn_cls_target: A Tensor with the target for each of the
                    anchors. The shape is [num_anchors,].
                rpn_bbox_target: A Tensor with the target for each of the
                    anchors. In case of ignoring the anchor for the target then
                    we still have a bbox target for each anchors, and it's
                    filled with zeroes when ignored.
        """
        # We start with a common conv layer applied to the feature map.
        self._instantiate_layers()
        self._proposal = RPNProposal(
            self._num_anchors, self._config.proposals, debug=self._debug
        )
        self._anchor_target = RPNTarget(
            self._num_anchors, self._config.target, seed=self._seed
        )

        prediction_dict = {}

        # Get the RPN feature using a simple conv net. Activation function
        # can be set to empty.
        rpn_conv_feature = self._rpn(conv_feature_map)
        rpn_feature = self._rpn_activation(rpn_conv_feature)

        # Then we apply separate conv layers for classification and regression.
        rpn_cls_score_original = self._rpn_cls(rpn_feature)
        rpn_bbox_pred_original = self._rpn_bbox(rpn_feature)
        # rpn_cls_score_original has shape (1, H, W, num_anchors * 2)
        # rpn_bbox_pred_original has shape (1, H, W, num_anchors * 4)
        # where H, W are height and width of the pretrained feature map.

        # Convert (flatten) `rpn_cls_score_original` which has two scalars per
        # anchor per location to be able to apply softmax.
        rpn_cls_score = tf.reshape(rpn_cls_score_original, [-1, 2])
        # Now that `rpn_cls_score` has shape (H * W * num_anchors, 2), we apply
        # softmax to the last dim.
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score)

        prediction_dict['rpn_cls_prob'] = rpn_cls_prob
        prediction_dict['rpn_cls_score'] = rpn_cls_score

        # Flatten bounding box delta prediction for easy manipulation.
        # We end up with `rpn_bbox_pred` having shape (H * W * num_anchors, 4).
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred_original, [-1, 4])

        prediction_dict['rpn_bbox_pred'] = rpn_bbox_pred

        # We have to convert bbox deltas to usable bounding boxes and remove
        # redundant ones using Non Maximum Suppression (NMS).
        proposal_prediction = self._proposal(
            rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape)

        prediction_dict['proposals'] = proposal_prediction['proposals']
        prediction_dict['scores'] = proposal_prediction['scores']

        if self._debug:
            prediction_dict['proposal_prediction'] = proposal_prediction

        if gt_boxes is not None:
            # When training we use a separate module to calculate the target
            # values we want to output.
            (rpn_cls_target, rpn_bbox_target,
             rpn_max_overlap) = self._anchor_target(
                all_anchors, gt_boxes, im_shape
            )

            prediction_dict['rpn_cls_target'] = rpn_cls_target
            prediction_dict['rpn_bbox_target'] = rpn_bbox_target

            if self._debug:
                prediction_dict['rpn_max_overlap'] = rpn_max_overlap
                variable_summaries(rpn_bbox_target, 'rpn_bbox_target', 'full')

        # Variables summaries.
        variable_summaries(prediction_dict['scores'], 'rpn_scores', 'reduced')
        variable_summaries(rpn_cls_prob, 'rpn_cls_prob', 'reduced')
        variable_summaries(rpn_bbox_pred, 'rpn_bbox_pred', 'reduced')

        if self._debug:
            variable_summaries(rpn_feature, 'rpn_feature', 'full')
            variable_summaries(
               rpn_cls_score_original, 'rpn_cls_score_original', 'full')
            variable_summaries(
               rpn_bbox_pred_original, 'rpn_bbox_pred_original', 'full')

            # Layer summaries.
            layer_summaries(self._rpn, 'full')
            layer_summaries(self._rpn_cls, 'full')
            layer_summaries(self._rpn_bbox, 'full')

        return prediction_dict

    def loss(self, prediction_dict):
        """
        Returns cost for Region Proposal Network based on:

        Args:
            rpn_cls_score: Score for being an object or not for each anchor
                in the image. Shape: (num_anchors, 2)
            rpn_cls_target: Ground truth labeling for each anchor. Should be
                * 1: for positive labels
                * 0: for negative labels
                * -1: for labels we should ignore.
                Shape: (num_anchors, )
            rpn_bbox_target: Bounding box output delta target for rpn.
                Shape: (num_anchors, 4)
            rpn_bbox_pred: Bounding box output delta prediction for rpn.
                Shape: (num_anchors, 4)
        Returns:
            Multiloss between cls probability and bbox target.
        """

        rpn_cls_score = prediction_dict['rpn_cls_score']
        rpn_cls_target = prediction_dict['rpn_cls_target']

        rpn_bbox_target = prediction_dict['rpn_bbox_target']
        rpn_bbox_pred = prediction_dict['rpn_bbox_pred']

        with tf.variable_scope('RPNLoss'):
            # Flatten already flat Tensor for usage as boolean mask filter.
            rpn_cls_target = tf.cast(tf.reshape(
                rpn_cls_target, [-1]), tf.int32, name='rpn_cls_target')
            # Transform to boolean tensor mask for not ignored.
            labels_not_ignored = tf.not_equal(
                rpn_cls_target, -1, name='labels_not_ignored')

            # Now we only have the labels we are going to compare with the
            # cls probability.
            labels = tf.boolean_mask(rpn_cls_target, labels_not_ignored)
            cls_score = tf.boolean_mask(rpn_cls_score, labels_not_ignored)

            # We need to transform `labels` to `cls_score` shape.
            # convert [1, 0] to [[0, 1], [1, 0]] for ce with logits.
            cls_target = tf.one_hot(labels, depth=2)

            # Equivalent to log loss
            ce_per_anchor = tf.nn.softmax_cross_entropy_with_logits(
                labels=cls_target, logits=cls_score
            )
            prediction_dict['cross_entropy_per_anchor'] = ce_per_anchor

            # Finally, we need to calculate the regression loss over
            # `rpn_bbox_target` and `rpn_bbox_pred`.
            # We use SmoothL1Loss.
            rpn_bbox_target = tf.reshape(rpn_bbox_target, [-1, 4])
            rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

            # We only care for positive labels (we ignore backgrounds since
            # we don't have any bounding box information for it).
            positive_labels = tf.equal(rpn_cls_target, 1)
            rpn_bbox_target = tf.boolean_mask(rpn_bbox_target, positive_labels)
            rpn_bbox_pred = tf.boolean_mask(rpn_bbox_pred, positive_labels)

            # We apply smooth l1 loss as described by the Fast R-CNN paper.
            reg_loss_per_anchor = smooth_l1_loss(
                rpn_bbox_pred, rpn_bbox_target, sigma=self._l1_sigma
            )

            prediction_dict['reg_loss_per_anchor'] = reg_loss_per_anchor

            # Loss summaries.
            tf.summary.scalar('batch_size', tf.shape(labels)[0], ['rpn'])
            foreground_cls_loss = tf.boolean_mask(
                ce_per_anchor, tf.equal(labels, 1))
            background_cls_loss = tf.boolean_mask(
                ce_per_anchor, tf.equal(labels, 0))
            tf.summary.scalar(
                'foreground_cls_loss',
                tf.reduce_mean(foreground_cls_loss), ['rpn'])
            tf.summary.histogram(
                'foreground_cls_loss', foreground_cls_loss, ['rpn'])
            tf.summary.scalar(
                'background_cls_loss',
                tf.reduce_mean(background_cls_loss), ['rpn'])
            tf.summary.histogram(
                'background_cls_loss', background_cls_loss, ['rpn'])
            tf.summary.scalar(
                'foreground_samples', tf.shape(rpn_bbox_target)[0], ['rpn'])

            return {
                'rpn_cls_loss': tf.reduce_mean(ce_per_anchor),
                'rpn_reg_loss': tf.reduce_mean(reg_loss_per_anchor),
            }
