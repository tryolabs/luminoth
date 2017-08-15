import numpy as np
import sonnet as snt
import tensorflow as tf

from .rcnn import RCNN
from .rpn import RPN

from luminoth.utils.anchors import generate_anchors_reference
from luminoth.utils.config import get_base_config
from luminoth.utils.vars import variable_summaries


class FasterRCNN(snt.AbstractModule):
    """Faster RCNN Network module

    Builds the Faster RCNN network architecture using different submodules.
    Calculates the total loss of the model based on the different losses by
    each of the submodules.

    It is also responsable for building the anchor reference which is used in
    graph for generating the dynamic anchors.

    """

    base_config = get_base_config(__file__)

    def __init__(self, config, debug=False, with_rcnn=True, num_classes=None,
                 name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)

        # Main configuration object, it holds not only the necessary
        # information for this module but also configuration for each of the
        # different submodules.
        self._config = config

        # Total number of classes to classify. If not using RCNN then it is not
        # used. TODO: Make it *more* optional.
        self._num_classes = config.network.num_classes

        # Generate network with RCNN thus allowing for classification of
        # objects and not just finding them.
        self._with_rcnn = config.network.with_rcnn

        # Turn on debug mode with returns more Tensors which can be used for
        # better visualization and (of course) debugging.
        self._debug = config.train.debug

        # Anchor config, check out the docs of base_config.yml for a better
        # understanding of how anchors work.
        self._anchor_base_size = config.anchors.base_size
        self._anchor_scales = np.array(config.anchors.scales)
        self._anchor_ratios = np.array(config.anchors.ratios)
        self._anchor_stride = config.anchors.stride

        # Anchor reference for building dynamic anchors for each image in the
        # computation graph.
        self._anchor_reference = generate_anchors_reference(
            self._anchor_base_size, self._anchor_ratios, self._anchor_scales
        )

        # Total number of anchors per point.
        self._num_anchors = self._anchor_reference.shape[0]

        # Weights used to sum each of the losses of the submodules
        self._rpn_cls_loss_weight = config.loss.rpn_cls_loss_weight
        self._rpn_reg_loss_weight = config.loss.rpn_reg_loss_weights

        self._rcnn_cls_loss_weight = config.loss.rcnn_cls_loss_weight
        self._rcnn_reg_loss_weight = config.loss.rcnn_reg_loss_weights
        self._losses_collections = ['fastercnn_losses']

    def _build(self, image, pretrained_feature_map, gt_boxes=None):
        """
        Returns bounding boxes and classification probabilities.

        Args:
            image: A tensor with the image.
                Its shape should be `(1, height, width, 3)`.
            pretrained_feature_map: A Tensor with the feature map for the image
                Its shape should be `(feature_height, feature_width, 512)`.
                The shape depends of the pretrained network in use.
            gt_boxes: A tensor with all the ground truth boxes of that image.
                Its shape should be `(num_gt_boxes, 4)`
                Where for each gt box we have (x1, y1, x2, y2), in that order.
            is_training: A boolean passed onto submodules from which it depends
                if targets for sub-minibatches are created or not.

        Returns:
            classification_prob: A tensor with the softmax probability for
                each of the bounding boxes found in the image.
                Its shape should be: (num_bboxes, num_categories + 1)
            classification_bbox: A tensor with the bounding boxes found.
                It's shape should be: (num_bboxes, 4). For each of the bboxes
                we have (x1, y1, x2, y2)
        """

        # The RPN submodule which generates proposals of objects.
        self._rpn = RPN(self._num_anchors, self._config.rpn, debug=self._debug)
        if self._with_rcnn:
            # The RCNN submodule which classifies RPN's proposals and
            # classifies them as background or a specific class.
            self._rcnn = RCNN(
                self._num_classes, self._config.rcnn, debug=self._debug
            )

        image_shape = tf.shape(image)[1:3]

        variable_summaries(
            pretrained_feature_map, 'pretrained_feature_map', ['rpn'])

        # Generate anchors for the image based on the anchor reference.
        all_anchors = self._generate_anchors(pretrained_feature_map)
        rpn_prediction = self._rpn(
            pretrained_feature_map, image_shape, all_anchors, gt_boxes=gt_boxes
        )

        prediction_dict = {
            'rpn_prediction': rpn_prediction,
        }

        if self._debug:
            prediction_dict['image'] = image
            prediction_dict['image_shape'] = image_shape
            prediction_dict['all_anchors'] = all_anchors
            prediction_dict['gt_boxes'] = gt_boxes

        if self._with_rcnn:
            classification_pred = self._rcnn(
                pretrained_feature_map, rpn_prediction['proposals'],
                image_shape, gt_boxes=gt_boxes
            )

            prediction_dict['classification_prediction'] = classification_pred

        return prediction_dict

    def loss(self, prediction_dict):
        """Compute the joint training loss for Faster RCNN.

        Args:
            prediction_dict: The output dictionary of the _build method from
                which we use to different main keys:

                rpn_prediction: A dictionary with the output Tensors from the
                    RPN.
                classification_prediction: A dictionary with the output Tensors
                    from the RCNN.
        """

        with tf.name_scope('losses'):
            rpn_loss_dict = self._rpn.loss(
                prediction_dict['rpn_prediction']
            )

            # Losses have a weight assigned, we multiply by them before saving
            # them.
            rpn_loss_dict['rpn_cls_loss'] = (
                rpn_loss_dict['rpn_cls_loss'] * self._rpn_cls_loss_weight)
            rpn_loss_dict['rpn_reg_loss'] = (
                rpn_loss_dict['rpn_reg_loss'] * self._rpn_reg_loss_weight)

            prediction_dict['rpn_loss_dict'] = rpn_loss_dict

            if self._with_rcnn:
                rcnn_loss_dict = self._rcnn.loss(
                    prediction_dict['classification_prediction']
                )

                rcnn_loss_dict['rcnn_cls_loss'] = (
                    rcnn_loss_dict['rcnn_cls_loss'] *
                    self._rcnn_cls_loss_weight
                )
                rcnn_loss_dict['rcnn_reg_loss'] = (
                    rcnn_loss_dict['rcnn_reg_loss'] *
                    self._rcnn_reg_loss_weight
                )

                prediction_dict['rcnn_loss_dict'] = rcnn_loss_dict
            else:
                rcnn_loss_dict = {}

            all_loses_items = (
                list(rpn_loss_dict.items()) + list(rcnn_loss_dict.items()))
            for loss_name, loss_tensor in all_loses_items:
                tf.summary.scalar(
                    loss_name, loss_tensor,
                    collections=self._losses_collections
                )
                # We add losses to the losses collection instead of manually
                # summing them just in case somebody wants to use it in another
                # place.
                tf.losses.add_loss(loss_tensor)

            # Regularization loss is automatically saved by TensorFlow, we log
            # it differently so we can visualize it independently.
            regularization_loss = tf.losses.get_regularization_loss()
            # Total loss without regularization
            no_reg_loss = tf.losses.get_total_loss(
                add_regularization_losses=False
            )
            total_loss = tf.losses.get_total_loss()

            tf.summary.scalar(
                'total_loss', total_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'no_reg_loss', no_reg_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'regularization_loss', regularization_loss,
                collections=self._losses_collections
            )

            # We return the total loss, which includes:
            # - rpn loss
            # - rcnn loss (if activated)
            # - regularization loss
            return total_loss

    def _generate_anchors(self, feature_map):
        """Generate anchor for an image.

        Using the feature map, the output of the pretrained network for an
        image, and the anchor_reference generated using the anchor config
        values. We generate a list of anchors.

        Anchors are just fixed bounding boxes of different ratios and sizes
        that are uniformly generated throught the image.

        Args:
            feature_map: A Tensor of shape
                `(feature_height, feature_width, 512)` using the VGG as
                pretrained.

        Returns:
            all_anchors: A flattened Tensor with all the anchors of shape
                `(num_anchors_per_points * feature_width * feature_height, 4)`
                using the (x1, y1, x2, y2) convention.
        """
        with tf.variable_scope('generate_anchors'):
            feature_map_shape = tf.shape(feature_map)[1:3]
            grid_width = feature_map_shape[1]
            grid_height = feature_map_shape[0]
            shift_x = tf.range(grid_width) * self._anchor_stride
            shift_y = tf.range(grid_height) * self._anchor_stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

            shift_x = tf.reshape(shift_x, [-1])
            shift_y = tf.reshape(shift_y, [-1])

            shifts = tf.stack(
                [shift_x, shift_y, shift_x, shift_y],
                axis=0
            )

            shifts = tf.transpose(shifts)
            # Shifts now is a (H x W, 4) Tensor

            num_anchors = self._anchor_reference.shape[0]
            num_anchor_points = tf.shape(shifts)[0]

            all_anchors = (
                self._anchor_reference.reshape((1, num_anchors, 4)) +
                tf.transpose(
                    tf.reshape(shifts, (1, num_anchor_points, 4)),
                    (1, 0, 2)
                )
            )

            all_anchors = tf.reshape(
                all_anchors, (num_anchors * num_anchor_points, 4)
            )
            return all_anchors

    @property
    def summary(self):
        """
        Generate merged summary of all the sub-summaries used inside the
        Faster R-CNN network.
        """
        summaries = [
            tf.summary.merge_all(key='rpn'),
        ]

        # if training:
        #    tf.summary.merge_all(key=self._losses_collections[0])

        if self._with_rcnn:
            summaries.append(tf.summary.merge_all(key='rcnn'))

        return tf.summary.merge(summaries)

    def get_trainable_vars(self):
        """Get trainable vars included in the module.
        """
        return snt.get_variables_in_module(self)
