import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from luminoth.models.base import SSDFeatureExtractor
from luminoth.models.ssd.ssd_proposal import SSDProposal
from luminoth.models.ssd.ssd_target import SSDTarget
from luminoth.models.ssd.ssd_utils import (
    generate_anchors_reference, adjust_bboxes
)
from luminoth.utils.losses import smooth_l1_loss
from luminoth.utils.vars import get_saver

from luminoth.utils.bbox_transform_tf import clip_boxes


DEFAULT_ENDPOINTS = {
    'ssd_1': (10, 10),
    'ssd_2': (5, 5),
    'ssd_3': (3, 3),
    'ssd_4': (1, 1),
}


class SSD(snt.AbstractModule):
    """TODO
    """

    def __init__(self, config, name='ssd'):
        super(SSD, self).__init__(name=name)

        # Main configuration object, it holds not only the necessary
        # information for this module but also configuration for each of the
        # different submodules.
        self._config = config

        # Total number of classes to classify.
        self._num_classes = config.model.network.num_classes

        # Turn on debug mode with returns more Tensors which can be used for
        # better visualization and (of course) debugging.
        self._debug = config.train.debug
        self._seed = config.train.seed

        # Anchor config, check out the docs of base_config.yml for a better
        # understanding of how anchors work.
        self._anchor_max_scale = config.model.anchors.max_scale
        self._anchor_min_scale = config.model.anchors.min_scale
        self._anchor_ratios = np.array(config.model.anchors.ratios)

        # Total number of anchors per point, per endpoint.
        self._anchors_per_point = config.model.anchors.anchors_per_point

        # Weight for the localization loss
        self._loc_loss_weight = config.model.loss.localization_loss_weight
        # TODO: why not use the default LOSSES collection?
        self._losses_collections = ['ssd_losses']

        # We want the pretrained model to be outside the ssd name scope.
        self.feature_extractor = SSDFeatureExtractor(
            config.model.base_network, parent_name=self.module_name
        )

    def _build(self, image, gt_boxes=None, is_training=True):
        """
        Returns bounding boxes and classification probabilities.

        Args:
            image: A tensor with the image.
                Its shape should be `(height, width, 3)`.
            gt_boxes: A tensor with all the ground truth boxes of that image.
                Its shape should be `(num_gt_boxes, 5)`
                Where for each gt box we have (x1, y1, x2, y2, label),
                in that order.
            is_training: A boolean to whether or not it is used for training.

        Returns:
            A dictionary with the following keys:
            predictions:
            proposal_prediction: A dictionary with:
                proposals: The proposals of the network after appling some
                    filters like negative area; and NMS
                proposals_label: A tensor with the label for each proposal.
                proposals_label_prob: A tensor with the softmax probability
                    for the label of each proposal.
            loc_pred: A tensor with the localization predictions
            cls_pred: A tensor with the classes predictions
        """
        if gt_boxes is not None:
            gt_boxes = tf.cast(gt_boxes, tf.float32)

        image_shape = (300, 300)  # TODO: get this from config
        image.set_shape((image_shape[0], image_shape[1], 3))
        image = tf.expand_dims(image, 0)  # TODO: batch size is hardcoded to 1
        feature_maps = self.feature_extractor(image, is_training=is_training)

        # Build a MultiBox predictor on top of each feature layer
        predictions = {}
        for i, (feature_map_name, feature_map) in enumerate(feature_maps.items()):
            num_anchors = self._anchors_per_point[i]

            # Location predictions
            num_loc_pred = num_anchors * 4
            loc_pred = slim.conv2d(feature_map, num_loc_pred, [3, 3],
                                   activation_fn=None,
                                   scope=feature_map_name + '/conv_loc',
                                   padding='SAME')
            loc_pred = tf.reshape(loc_pred, [-1, 4])

            # Class predictions
            num_cls_pred = num_anchors * (self._num_classes + 1)
            cls_pred = slim.conv2d(feature_map, num_cls_pred, [3, 3],
                                   activation_fn=None,
                                   scope=feature_map_name + '/conv_cls',
                                   padding='SAME')
            cls_pred = tf.reshape(cls_pred, [-1, self._num_classes + 1])

            predictions[feature_map_name] = {}
            predictions[feature_map_name]['loc_pred'] = loc_pred
            predictions[feature_map_name]['cls_pred'] = cls_pred
            predictions[feature_map_name]['prob'] = slim.softmax(cls_pred)


        self.anchors = self.generate_anchors(feature_maps)

        # Get all_anchors from each endpoint
        all_anchors_list = []
        loc_pred_list = []
        cls_pred_list = []
        cls_prob_list = []
        # for ind, endpoint in enumerate(self._endpoints):
        for i, (layer_name, layer) in enumerate(feature_maps.items):
            adjusted_bboxes = adjust_bboxes(
                self.anchors[endpoint],
                tf.cast(self._endpoints_outputs[ind][0], tf.float32),
                tf.cast(self._endpoints_outputs[ind][1], tf.float32),
                tf.cast(tf.shape(image)[1], tf.float32),
                tf.cast(tf.shape(image)[2], tf.float32)
            )
            # Clip anchors to the image.
            adjusted_bboxes = clip_boxes(
                adjusted_bboxes, tf.cast(tf.shape(image)[1:3], tf.int32))
            all_anchors_list.append(adjusted_bboxes)
            loc_pred_list.append(predictions[endpoint]['loc_pred'])
            cls_prob_list.append(
                slim.softmax(predictions[endpoint]['cls_pred']))
            cls_pred_list.append(predictions[endpoint]['cls_pred'])
        all_anchors = tf.concat(all_anchors_list, axis=0)
        loc_pred = tf.concat(loc_pred_list, axis=0)
        cls_pred = tf.concat(cls_pred_list, axis=0)
        cls_prob = tf.concat(cls_prob_list, axis=0)

        prediction_dict = {}
        all_anchors_target = all_anchors
        if gt_boxes is not None:
            # Get the targets and returns it
            self._target = SSDTarget(self._num_classes, all_anchors.shape[0],
                                     self._config.model.target)

            proposals_label_target, bbox_offsets_target = self._target(
                cls_prob, all_anchors, gt_boxes,
                tf.cast(tf.shape(image), tf.float32)
            )

            if is_training:
                with tf.name_scope('prepare_batch'):
                    # We flatten to set shape, but it is already a flat Tensor.
                    in_batch_proposals = tf.reshape(
                        tf.greater_equal(proposals_label_target, 0), [-1]
                    )
                    all_anchors_target = tf.boolean_mask(
                        all_anchors, in_batch_proposals)
                    bbox_offsets_target = tf.boolean_mask(
                        bbox_offsets_target, in_batch_proposals)
                    proposals_label_target = tf.boolean_mask(
                        proposals_label_target, in_batch_proposals)
                    cls_pred = tf.boolean_mask(
                        cls_pred, in_batch_proposals)
                    cls_prob = tf.boolean_mask(
                        cls_prob, in_batch_proposals)
                    loc_pred = tf.boolean_mask(
                        loc_pred, in_batch_proposals)

            prediction_dict['target'] = {
                'cls': proposals_label_target,
                'bbox_offsets': bbox_offsets_target,
            }

        # Get the proposals and save the result
        self._proposal = SSDProposal(all_anchors_target.shape[0],
                                     self._num_classes,
                                     self._config.model.proposals,
                                     debug=self._debug)
        proposal_prediction = self._proposal(
            cls_prob, loc_pred, all_anchors_target,
            tf.cast(tf.shape(image)[1:3], tf.float32)
        )

        prediction_dict.update({
            'predictions': predictions,
            'proposal_prediction': proposal_prediction,
            'classification_prediction': proposal_prediction,
            'cls_pred': cls_pred,
            'loc_pred': loc_pred
        })

        # TODO add variable summaries

        if self._debug:
            prediction_dict['anchors'] = self.anchors
            prediction_dict['all_anchors'] = all_anchors
            prediction_dict['all_anchors_target'] = all_anchors_target
            prediction_dict['cls_prob'] = cls_prob

        return prediction_dict

    def loss(self, prediction_dict):
        """Compute the loss for SSD.

        Args:
            prediction_dict: The output dictionary of the _build method from
                which we use different main keys:

                cls_pred: A dictionary with the classes classification.
                loc_pred: A dictionary with the localization predictions
                target: A dictionary with the targets for both classes and
                    localizations.

        Returns:
            A tensor for the total loss.
        """

        with tf.name_scope('losses'):

            cls_pred = prediction_dict['cls_pred']
            cls_target = tf.cast(
                prediction_dict['target']['cls'], tf.int32
            )

            # We only care for the targets that are >= 0
            not_ignored = tf.reshape(tf.greater_equal(
                cls_target, 0), [-1], name='not_ignored')
            # We apply boolean mask to score and target.
            cls_pred_labeled = tf.boolean_mask(
                cls_pred, not_ignored, name='cls_pred_labeled')
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
                    labels=cls_target_one_hot, logits=cls_pred_labeled
                )
            )
            # Second we need to calculate the smooth l1 loss between
            # `bbox_offsets` and `bbox_offsets_target`.
            bbox_offsets = prediction_dict['loc_pred']
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

            # Calculate the smooth l1 loss between the flatten bboxes
            # offsets  and the labeled targets.
            # TODO: reg loss can be confused between regularization and
            #       regression, rename
            reg_loss_per_proposal = smooth_l1_loss(
                bbox_offsets_labeled, bbox_offsets_target_labeled)

            cls_loss = tf.reduce_sum(cross_entropy_per_proposal)
            bbox_loss = tf.reduce_sum(reg_loss_per_proposal)

            # Following the paper, set loss to 0. if there are 0 bboxes
            # assigned as foreground targets.
            safety_condition = tf.not_equal(
                tf.shape(bbox_offsets_labeled)[0], 0
            )
            final_loss = tf.cond(
                safety_condition,
                true_fn=lambda: (
                    (cls_loss + bbox_loss * self._loc_loss_weight) /
                    tf.cast(tf.shape(bbox_offsets_labeled)[0], tf.float32)
                ),
                false_fn=lambda: 0.0
            )
            tf.losses.add_loss(final_loss)
            total_loss = tf.losses.get_total_loss()

            tf.summary.scalar(
                'total_loss', total_loss,
                collections=self._losses_collections
            )

            return total_loss

    def generate_anchors_per_endpoint(self):
        """
        Returns a dictionary containing the anchors per endpoint.

        Returns:
        anchors: A dictionary with `endpoints` as keys and an array of anchors
            as values ('[[x_min, y_min, x_max, y_max], ...]') with shape
            (anchors_per_point[i] * endpoints_outputs[i][0]
             * endpoints_outputs[i][1], 4)
        """
        # Calculate the scales (usign scale min/max and number of endpoints).
        num_endpoints = len(self._endpoints)
        scales = np.zeros([num_endpoints])
        for endpoint in range(num_endpoints):
            scales[endpoint] = (
                self._anchor_min_scale +
                (self._anchor_max_scale - self._anchor_min_scale) *
                (endpoint) / (num_endpoints - 1)
            )

        # For each endpoint calculate the anchors with the appropiate size.
        anchors = {}
        for ind, endpoint in enumerate(self._endpoints):
            # Get the anchors reference for this endpoint
            anchor_reference = generate_anchors_reference(
                self._anchor_ratios, scales[ind: ind + 2],
                self._anchors_per_point[ind],
                self._endpoints_outputs[ind]
            )
            anchors[endpoint] = self._generate_anchors(
                self._endpoints_outputs[ind], anchor_reference)

        return anchors

    def _generate_anchors(self, feature_map_shape, anchor_reference):
        """Generate anchor for an image.

        Using the feature map, the output of the pretrained network for an
        image, and the anchor_reference generated using the anchor config
        values. We generate a list of anchors.

        Anchors are just fixed bounding boxes of different ratios and sizes
        that are uniformly generated throught the image.

        Args:
            feature_map_shape: Shape of the convolutional feature map used as
                input for the RPN. Should be (batch, height, width, depth).

        Returns:
            all_anchors: A flattened Tensor with all the anchors of shape
                `(num_anchors_per_points * feature_width * feature_height, 4)`
                using the (x1, y1, x2, y2) convention.
        """
        with tf.variable_scope('generate_anchors'):
            grid_width = feature_map_shape[1]  # width
            grid_height = feature_map_shape[0]  # height
            shift_x = tf.range(grid_width)
            shift_y = tf.range(grid_height)
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

            shift_x = tf.reshape(shift_x, [-1])
            shift_y = tf.reshape(shift_y, [-1])

            shifts = tf.stack(
                [shift_x, shift_y, shift_x, shift_y],
                axis=0
            )

            shifts = tf.transpose(shifts)
            # Shifts now is a (H x W, 4) Tensor

            # Expand dims to use broadcasting sum.
            all_anchors = (
                tf.expand_dims(anchor_reference, axis=0) +
                tf.cast(tf.expand_dims(shifts, axis=1), tf.float64)
            )

            # Flatten
            all_anchors = tf.reshape(
                all_anchors, (-1, 4)
            )
            return all_anchors

    @property
    def summary(self):
        """
        Generate merged summary of all the sub-summaries used inside the
        ssd network.
        """
        summaries = [
            tf.summary.merge_all(key=self._losses_collections[0])
        ]

        return tf.summary.merge(summaries)

    def get_trainable_vars(self):
        """Get trainable vars included in the module.
        """
        trainable_vars = snt.get_variables_in_module(self)
        if self._config.model.base_network.trainable:
            pretrained_trainable_vars = self.base_network.get_trainable_vars()
            tf.logging.info('Training {} vars from pretrained module.'.format(
                len(pretrained_trainable_vars)))
            trainable_vars += pretrained_trainable_vars
        else:
            tf.logging.info('Not training variables from pretrained module')

        return trainable_vars

    def get_saver(self, ignore_scope=None):
        """Get an instance of tf.train.Saver for all modules and submodules.
        """
        return get_saver((self, self.base_network), ignore_scope=ignore_scope)

    def load_pretrained_weights(self):
        """Get operation to load pretrained weights from file.
        """
        with tf.control_dependencies([tf.global_variables_initializer()]):
            res = self.base_network.load_weights()
        return res
