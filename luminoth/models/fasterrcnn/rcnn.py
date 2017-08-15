import sonnet as snt
import tensorflow as tf

from luminoth.models.fasterrcnn.rcnn_proposal import RCNNProposal
from luminoth.models.fasterrcnn.rcnn_target import RCNNTarget
from luminoth.models.fasterrcnn.roi_pool import ROIPoolingLayer
from luminoth.utils.losses import smooth_l1_loss
from luminoth.utils.vars import variable_summaries, get_initializer


class RCNN(snt.AbstractModule):
    """RCNN: Region-based Convolutional Neural Network

    Given a number of proposals (bounding boxes on an image) and a feature map
    of that image, RCNN adjust the bounding box and classifies the region as
    either background or a specific class.

    These are the steps for classifying the images:
        - Region of Interest Pooling. It extracts features from the feature map
        based on the proposals into a fixed size (applying extrapolation).
        - Uses two fully connected layers to generate a smaller Tensor for each
        region
        - Finally, a fully conected layer for classifying the class (adding
        background as a possible class), and a regression for adjusting the
        bounding box, one 4-d regression for each one of the possible classes.
        - Using the class probability and the corresponding bounding box
        regression it, in case of not classifying the region as background it
        generates the final object bounding box with class and class
        probability assigned to it.
    """

    def __init__(self, num_classes, config, debug=False, name='rcnn'):
        super(RCNN, self).__init__(name=name)
        self._num_classes = num_classes
        # List of the fully connected layer sized used before classifying and
        # adjusting the bounding box.
        self._layer_sizes = config.layer_sizes
        self._activation = tf.nn.relu6
        self._dropout_keep_prob = config.dropout_keep_prop

        self.initializer = get_initializer(config.initializer)
        self.regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.l2_regularization_scale)

        # Debug mode makes the module return more detailed Tensors which can be
        # useful for debugging.
        self._debug = debug
        self._config = config

    def _instantiate_layers(self):
        # We define layers as an array since they are simple fully conected
        # ones and it should be easy to tune it from the network config.
        self._layers = [
            snt.Linear(
                layer_size,
                name="fc_{}".format(i),
                initializers={'w': self.initializer},
                regularizers={'w': self.regularizer},
            )
            for i, layer_size in enumerate(self._layer_sizes)
        ]

        # We define the classifier layer having a num_classes + 1 background
        # since we want to be able to predict if the proposal is background as
        # well.
        self._classifier_layer = snt.Linear(
            self._num_classes + 1, name="fc_classifier",
            initializers={'w': self.initializer},
            regularizers={'w': self.regularizer},
        )

        # The bounding box adjustment layer has 4 times the number of classes
        # We choose which to use depending on the output of the classifier
        # layer
        self._bbox_layer = snt.Linear(
            self._num_classes * 4, name="fc_bbox",
            initializers={'w': self.initializer},
            regularizers={'w': self.regularizer}
        )

        # ROIPoolingLayer is used to extract the feature from the feature map
        # using the proposals.
        self._roi_pool = ROIPoolingLayer(self._config.roi, debug=self._debug)
        # RCNNTarget is used to define a minibatch and the correct values for
        # each of the proposals.
        self._rcnn_target = RCNNTarget(
            self._num_classes, self._config.target, debug=self._debug
        )
        # RCNNProposal generates the final bounding boxes and tries to remove
        # duplicates.
        self._rcnn_proposal = RCNNProposal(
            self._num_classes, self._config.proposals
        )

    def _build(self, pretrained_feature_map, proposals, im_shape,
               gt_boxes=None):
        """
        Classifies proposals based on the pooled feature map.

        Args:
            pretrained_feature_map: The feature map of the image extracted
                using the pretrained network.
                Shape (num_proposals, pool_height, pool_width, 512).
            proposals: A Tensor with the bounding boxes proposed by de RPN. Its
                shape is (total_num_proposals, 4) using the encoding
                (x1, y1, x2, y2).
            im_shape: A Tensor with the shape of the image in the form of
                (image_height, image_width)
            inference: Whether the module is created for training or not,
                which in turn defines if targets are calculated.
            gt_boxes (optional): A Tensor with the ground truth boxes of the
                image. Its shape is (total_num_gt, 5), using the encoding
                (x1, y1, x2, y2, label).

        Returns:
            prediction_dict a dict with the object predictions.

        TODO: Not training behaviour, we have to extract all ROI and classify
        them?
        """
        self._instantiate_layers()

        prediction_dict = {}

        if gt_boxes is not None:
            # TODO: If not training, we should not depend on proposals_target
            # to use region of interest pooling, we would just need to use all
            # available proposals.
            proposals_target, bbox_target = self._rcnn_target(
                proposals, gt_boxes)

            with tf.name_scope('prepare_batch'):
                # We flatten to set shape, but it is already a flat Tensor.
                in_batch_proposals = tf.reshape(
                    tf.greater_equal(proposals_target, 0), [-1]
                )
                roi_proposals = tf.boolean_mask(
                    proposals, in_batch_proposals)
                roi_bbox_target = tf.boolean_mask(
                    bbox_target, in_batch_proposals)
                roi_proposals_target = tf.boolean_mask(
                    proposals_target, in_batch_proposals)

            prediction_dict['cls_target'] = roi_proposals_target
            prediction_dict['bbox_offsets_target'] = roi_bbox_target

        else:
            roi_proposals = proposals

        roi_prediction = self._roi_pool(
            roi_proposals, pretrained_feature_map,
            im_shape
        )

        if self._debug:
            # Save raw roi prediction in debug mode.
            prediction_dict['roi_prediction'] = roi_prediction

        pooled_layer = roi_prediction['roi_pool']

        # We treat num proposals as batch number so that when flattening we
        # get a (num_proposals, flatten_pooled_feature_map_size) Tensor.
        flatten_net = tf.contrib.layers.flatten(pooled_layer)
        net = tf.identity(flatten_net)

        if self._debug:
            prediction_dict['flatten_net'] = net
            variable_summaries(pooled_layer, 'pooled_layer', ['rcnn'])

        # After flattening we are lef with a
        # (num_proposals, pool_height * pool_width * 512) Tensor.
        # The first dimension works as batch size when applied to snt.Linear.
        for i, layer in enumerate(self._layers):
            # Through FC layer.
            net = layer(net)
            if self._debug:
                prediction_dict['layer_{}_out'.format(i)] = net

            # Apply activation and dropout.
            net = self._activation(net)
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        cls_score = self._classifier_layer(net)
        prob = tf.nn.softmax(cls_score, dim=1)
        bbox_offsets = self._bbox_layer(net)

        # Get final objects proposals based on the probabilty, the offsets and
        # the original proposals.
        proposal_prediction = self._rcnn_proposal(
            roi_proposals, bbox_offsets, prob, im_shape)

        variable_summaries(prob, 'prob', ['rcnn'])
        variable_summaries(bbox_offsets, 'bbox_offsets', ['rcnn'])

        prediction_dict['cls_score'] = cls_score
        prediction_dict['cls_prob'] = prob
        prediction_dict['bbox_offsets'] = bbox_offsets
        prediction_dict['roi_proposals'] = roi_proposals

        # objects, objects_labels, and objects_labels_prob are the only keys
        # that matter for drawing objects.
        prediction_dict['objects'] = proposal_prediction['objects']
        prediction_dict['objects_labels'] = (
            proposal_prediction['proposal_label'])
        prediction_dict['objects_labels_prob'] = (
            proposal_prediction['proposal_label_prob'])

        if self._debug:
            prediction_dict['proposal_prediction'] = proposal_prediction

        return prediction_dict

    def loss(self, prediction_dict):
        """
        Returns cost for RCNN based on:

        Args:
            prediction_dict with keys:
                cls_score: shape (num_proposals, num_classes + 1)
                    Has the class scoring for each the proposals. Classes are
                    1-indexed with 0 being the background.

                cls_prob: shape (num_proposals, num_classes + 1)
                    Application of softmax on cls_score.

                cls_target: shape (num_proposals,)
                    Has the correct label for each of the proposals.
                    0 => background
                    1..n => 1-indexed classes

                bbox_offsets: shape (num_proposals, num_classes * 4)
                    Has the offset for each proposal for each class.
                    We have to compare only the proposals labeled with the
                    offsets for that label.

                bbox_offsets_target: shape (num_proposals, 4)
                    Has the true offset of each proposal for the true label.
                    In case of not having a true label (non-background) then
                    it's just zeroes.

        Returns:
            loss_dict with keys:
                rcnn_cls_loss: The cross-entropy or log-loss of the
                    classification tasks between then num_classes + background.
                rcnn_reg_loss: The smooth L1 loss for the bounding box
                    regression task to adjust correctly labeled boxes.

        """
        with self._enter_variable_scope():
            with tf.name_scope('RCNNLoss'):
                cls_score = prediction_dict['cls_score']
                # cls_prob = prediction_dict['cls_prob']
                # Cast target explicitly as int32.
                cls_target = tf.cast(prediction_dict['cls_target'], tf.int32)

                # First we need to calculate the log loss betweetn cls_prob and
                # cls_target

                # We only care for the targets that are >= 0
                not_ignored = tf.reshape(tf.greater_equal(
                    cls_target, 0), [-1], name='not_ignored')
                # We apply boolean mask to score, prob and target.
                cls_score_labeled = tf.boolean_mask(
                    cls_score, not_ignored, name='cls_score_labeled')
                # cls_prob_labeled = tf.boolean_mask(
                #    cls_prob, not_ignored, name='cls_prob_labeled')
                cls_target_labeled = tf.boolean_mask(
                    cls_target, not_ignored, name='cls_target_labeled')

                # Transform to one-hot vector
                cls_target_one_hot = tf.one_hot(
                    cls_target_labeled, depth=self._num_classes + 1,
                    name='cls_target_one_hot'
                )

                # We get cross entropy loss of each proposal.
                cross_entropy_per_proposal = (
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=cls_target_one_hot, logits=cls_score_labeled
                    )
                )

                if self._debug:
                    # Save the cross entropy per proposal to be able to
                    # visualize proposals with high and low error.
                    prediction_dict['cross_entropy_per_proposal'] = (
                        cross_entropy_per_proposal
                    )

                # Second we need to calculate the smooth l1 loss between
                # `bbox_offsets` and `bbox_offsets_target`.
                bbox_offsets = prediction_dict['bbox_offsets']
                bbox_offsets_target = prediction_dict['bbox_offsets_target']

                # We only want the non-background labels bounding boxes.
                not_ignored = tf.reshape(tf.greater(cls_target, 0), [-1])
                bbox_offsets_labeled = tf.boolean_mask(
                    bbox_offsets, not_ignored, name='bbox_offsets_labeled')
                bbox_offsets_target_labeled = tf.boolean_mask(
                    bbox_offsets_target, not_ignored,
                    name='bbox_offsets_target_labeled'
                )

                cls_target_labeled = tf.boolean_mask(
                    cls_target, not_ignored, name='cls_target_labeled')
                # `cls_target_labeled` is based on `cls_target` which has
                # `num_classes` + 1 classes.
                # for making `one_hot` with depth `num_classes` to work we need
                # to lower them to make them 0-index.
                cls_target_labeled = cls_target_labeled - 1

                cls_target_one_hot = tf.one_hot(
                    cls_target_labeled, depth=self._num_classes,
                    name='cls_target_one_hot'
                )

                # cls_target now is (num_labeled, num_classes)
                bbox_flatten = tf.reshape(
                    bbox_offsets_labeled, [-1, 4], name='bbox_flatten')

                # We use the flatten cls_target_one_hot as boolean mask for the
                # bboxes.
                cls_flatten = tf.cast(tf.reshape(
                    cls_target_one_hot, [-1]), tf.bool, 'cls_flatten_as_bool')

                bbox_offset_cleaned = tf.boolean_mask(
                    bbox_flatten, cls_flatten, 'bbox_offset_cleaned')

                # Calculate the smooth l1 loss between the "cleaned" bboxes
                # offsets (that means, the useful results) and the labeled
                # targets.
                reg_loss_per_proposal = smooth_l1_loss(
                    bbox_offset_cleaned, bbox_offsets_target_labeled)

                tf.summary.scalar(
                    'rcnn_foreground_samples',
                    tf.shape(bbox_offset_cleaned)[0], ['rcnn']
                )

                if self._debug:
                    # Also save reg loss per proposals to be able to visualize
                    # good and bad proposals in debug mode.
                    prediction_dict['reg_loss_per_proposal'] = (
                        reg_loss_per_proposal
                    )

                return {
                    'rcnn_cls_loss': tf.reduce_mean(
                        cross_entropy_per_proposal
                    ),
                    'rcnn_reg_loss': tf.reduce_mean(reg_loss_per_proposal),
                }
