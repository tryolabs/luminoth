import sonnet as snt
import tensorflow as tf

from luminoth.models.fasterrcnn.rcnn_proposal import RCNNProposal
from luminoth.models.fasterrcnn.rcnn_target import RCNNTarget
from luminoth.models.fasterrcnn.roi_pool import ROIPoolingLayer
from luminoth.utils.losses import smooth_l1_loss
from luminoth.utils.vars import (
    get_initializer, layer_summaries, variable_summaries,
    get_activation_function
)


class RCNN(snt.AbstractModule):
    """RCNN: Region-based Convolutional Neural Network.

    Given region proposals (bounding boxes on an image) and a feature map of
    that image, RCNN adjusts the bounding boxes and classifies each region as
    either background or a specific object class.

    Steps:
        1. Region of Interest Pooling. Extract features from the feature map
           (based on the proposals) and convert into fixed size tensors
           (applying extrapolation).
        2. Two fully connected layers generate a smaller tensor for each
           region.
        3. A fully conected layer outputs the probability distribution over the
           classes (plus a background class), and another fully connected layer
           outputs the bounding box regressions (one 4-d regression for each of
           the possible classes).

    Using the class probability, filter regions classified as background. For
    the remaining regions, use the class probability together with the
    corresponding bounding box regression offsets to generate the final object
    bounding boxes, with classes and probabilities assigned.
    """

    def __init__(self, num_classes, config, debug=False, seed=None,
                 name='rcnn'):
        super(RCNN, self).__init__(name=name)
        self._num_classes = num_classes
        # List of the fully connected layer sizes used before classifying and
        # adjusting the bounding box.
        self._layer_sizes = config.layer_sizes
        self._activation = get_activation_function(config.activation_function)
        self._dropout_keep_prob = config.dropout_keep_prob
        self._use_mean = config.use_mean

        self._rcnn_initializer = get_initializer(
            config.rcnn_initializer, seed=seed
        )
        self._cls_initializer = get_initializer(
            config.cls_initializer, seed=seed
        )
        self._bbox_initializer = get_initializer(
            config.bbox_initializer, seed=seed
        )
        self.regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.l2_regularization_scale)

        self._l1_sigma = config.l1_sigma

        # Debug mode makes the module return more detailed Tensors which can be
        # useful for debugging.
        self._debug = debug
        self._config = config
        self._seed = seed

    def _instantiate_layers(self):
        # We define layers as an array since they are simple fully connected
        # ones and it should be easy to tune it from the network config.
        self._layers = [
            snt.Linear(
                layer_size,
                name='fc_{}'.format(i),
                initializers={'w': self._rcnn_initializer},
                regularizers={'w': self.regularizer},
            )
            for i, layer_size in enumerate(self._layer_sizes)
        ]
        # We define the classifier layer having a num_classes + 1 background
        # since we want to be able to predict if the proposal is background as
        # well.
        self._classifier_layer = snt.Linear(
            self._num_classes + 1, name='fc_classifier',
            initializers={'w': self._cls_initializer},
            regularizers={'w': self.regularizer},
        )

        # The bounding box adjustment layer has 4 times the number of classes
        # We choose which to use depending on the output of the classifier
        # layer
        self._bbox_layer = snt.Linear(
            self._num_classes * 4, name='fc_bbox',
            initializers={'w': self._bbox_initializer},
            regularizers={'w': self.regularizer}
        )

        # ROIPoolingLayer is used to extract the feature from the feature map
        # using the proposals.
        self._roi_pool = ROIPoolingLayer(self._config.roi, debug=self._debug)
        # RCNNTarget is used to define a minibatch and the correct values for
        # each of the proposals.
        self._rcnn_target = RCNNTarget(
            self._num_classes, self._config.target, seed=self._seed
        )
        # RCNNProposal generates the final bounding boxes and tries to remove
        # duplicates.
        self._rcnn_proposal = RCNNProposal(
            self._num_classes, self._config.proposals
        )

    def _build(self, conv_feature_map, proposals, im_shape, base_network,
               gt_boxes=None, is_training=False):
        """
        Classifies & refines proposals based on the pooled feature map.

        Args:
            conv_feature_map: The feature map of the image, extracted
                using the pretrained network.
                Shape: (num_proposals, pool_height, pool_width, 512).
            proposals: A Tensor with the bounding boxes proposed by the RPN.
                Shape: (total_num_proposals, 4).
                Encoding: (x1, y1, x2, y2).
            im_shape: A Tensor with the shape of the image in the form of
                (image_height, image_width).
            gt_boxes (optional): A Tensor with the ground truth boxes of the
                image.
                Shape: (total_num_gt, 5).
                Encoding: (x1, y1, x2, y2, label).
            is_training (optional): A boolean to determine if we are just using
                the module for training or just inference.

        Returns:
            prediction_dict: a dict with the object predictions.
                It should have the keys:
                objects:
                labels:
                probs:

                rcnn:
                target:

        """
        self._instantiate_layers()

        prediction_dict = {'_debug': {}}

        if gt_boxes is not None:
            proposals_target, bbox_offsets_target = self._rcnn_target(
                proposals, gt_boxes)

            if is_training:
                with tf.name_scope('prepare_batch'):
                    # We flatten to set shape, but it is already a flat Tensor.
                    in_batch_proposals = tf.reshape(
                        tf.greater_equal(proposals_target, 0), [-1]
                    )
                    proposals = tf.boolean_mask(
                        proposals, in_batch_proposals)
                    bbox_offsets_target = tf.boolean_mask(
                        bbox_offsets_target, in_batch_proposals)
                    proposals_target = tf.boolean_mask(
                        proposals_target, in_batch_proposals)

            prediction_dict['target'] = {
                'cls': proposals_target,
                'bbox_offsets': bbox_offsets_target,
            }

        roi_prediction = self._roi_pool(proposals, conv_feature_map, im_shape)

        if self._debug:
            # Save raw roi prediction in debug mode.
            prediction_dict['_debug']['roi'] = roi_prediction

        pooled_features = roi_prediction['roi_pool']
        features = base_network._build_tail(
            pooled_features, is_training=is_training
        )

        if self._use_mean:
            # We avg our height and width dimensions for a more
            # "memory-friendly" Tensor.
            features = tf.reduce_mean(features, [1, 2])

        # We treat num proposals as batch number so that when flattening we
        # get a (num_proposals, flatten_pooled_feature_map_size) Tensor.
        flatten_features = tf.contrib.layers.flatten(features)
        net = tf.identity(flatten_features)

        if is_training:
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        if self._debug:
            prediction_dict['_debug']['flatten_net'] = net

        # After flattening we are left with a Tensor of shape
        # (num_proposals, pool_height * pool_width * 512).
        # The first dimension works as batch size when applied to snt.Linear.
        for i, layer in enumerate(self._layers):
            # Through FC layer.
            net = layer(net)

            # Apply activation and dropout.
            variable_summaries(
                net, 'fc_{}_preactivationout'.format(i), 'reduced'
            )
            net = self._activation(net)
            if self._debug:
                prediction_dict['_debug']['layer_{}_out'.format(i)] = net

            variable_summaries(net, 'fc_{}_out'.format(i), 'reduced')
            if is_training:
                net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        cls_score = self._classifier_layer(net)
        cls_prob = tf.nn.softmax(cls_score, dim=1)
        bbox_offsets = self._bbox_layer(net)

        prediction_dict['rcnn'] = {
            'cls_score': cls_score,
            'cls_prob': cls_prob,
            'bbox_offsets': bbox_offsets,
        }

        # Get final objects proposals based on the probabilty, the offsets and
        # the original proposals.
        proposals_pred = self._rcnn_proposal(
            proposals, bbox_offsets, cls_prob, im_shape)

        # objects, objects_labels, and objects_labels_prob are the only keys
        # that matter for drawing objects.
        prediction_dict['objects'] = proposals_pred['objects']
        prediction_dict['labels'] = proposals_pred['proposal_label']
        prediction_dict['probs'] = proposals_pred['proposal_label_prob']

        if self._debug:
            prediction_dict['_debug']['proposal'] = proposals_pred

        # Calculate summaries for results
        variable_summaries(cls_prob, 'cls_prob', 'reduced')
        variable_summaries(bbox_offsets, 'bbox_offsets', 'reduced')

        if self._debug:
            variable_summaries(pooled_features, 'pooled_features', 'full')
            layer_summaries(self._classifier_layer, 'full')
            layer_summaries(self._bbox_layer, 'full')

        return prediction_dict

    def loss(self, prediction_dict):
        """
        Returns cost for RCNN based on:

        Args:
            prediction_dict with keys:
                rcnn:
                    cls_score: shape (num_proposals, num_classes + 1)
                        Has the class scoring for each the proposals. Classes
                        are 1-indexed with 0 being the background.

                    cls_prob: shape (num_proposals, num_classes + 1)
                        Application of softmax on cls_score.

                    bbox_offsets: shape (num_proposals, num_classes * 4)
                        Has the offset for each proposal for each class.
                        We have to compare only the proposals labeled with the
                        offsets for that label.

                target:
                    cls_target: shape (num_proposals,)
                        Has the correct label for each of the proposals.
                        0 => background
                        1..n => 1-indexed classes

                    bbox_offsets_target: shape (num_proposals, 4)
                        Has the true offset of each proposal for the true
                        label.
                        In case of not having a true label (non-background)
                        then it's just zeroes.

        Returns:
            loss_dict with keys:
                rcnn_cls_loss: The cross-entropy or log-loss of the
                    classification tasks between then num_classes + background.
                rcnn_reg_loss: The smooth L1 loss for the bounding box
                    regression task to adjust correctly labeled boxes.

        """
        with tf.name_scope('RCNNLoss'):
            cls_score = prediction_dict['rcnn']['cls_score']
            # cls_prob = prediction_dict['rcnn']['cls_prob']
            # Cast target explicitly as int32.
            cls_target = tf.cast(
                prediction_dict['target']['cls'], tf.int32
            )

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

            tf.summary.scalar(
                'batch_size',
                tf.shape(cls_score_labeled)[0], ['rcnn']
            )

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
                prediction_dict['_debug']['losses'] = {}
                # Save the cross entropy per proposal to be able to
                # visualize proposals with high and low error.
                prediction_dict['_debug']['losses'][
                    'cross_entropy_per_proposal'
                ] = (
                    cross_entropy_per_proposal
                )

            # Second we need to calculate the smooth l1 loss between
            # `bbox_offsets` and `bbox_offsets_target`.
            bbox_offsets = prediction_dict['rcnn']['bbox_offsets']
            bbox_offsets_target = (
                prediction_dict['target']['bbox_offsets']
            )

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
                bbox_offset_cleaned, bbox_offsets_target_labeled,
                sigma=self._l1_sigma
            )

            tf.summary.scalar(
                'rcnn_foreground_samples',
                tf.shape(bbox_offset_cleaned)[0], ['rcnn']
            )

            if self._debug:
                # Also save reg loss per proposals to be able to visualize
                # good and bad proposals in debug mode.
                prediction_dict['_debug']['losses'][
                    'reg_loss_per_proposal'
                ] = (
                    reg_loss_per_proposal
                )

            return {
                'rcnn_cls_loss': tf.reduce_mean(cross_entropy_per_proposal),
                'rcnn_reg_loss': tf.reduce_mean(reg_loss_per_proposal),
            }
