import numpy as np
import sonnet as snt
import tensorflow as tf

from .rcnn import RCNN
from .rpn import RPN

from luminoth.utils.generate_anchors import generate_anchors as generate_anchors_ref
from luminoth.utils.ops import meshgrid
from luminoth.utils.image import draw_bboxes
from luminoth.utils.vars import variable_summaries
from luminoth.utils.config import get_base_config


class FasterRCNN(snt.AbstractModule):
    """Faster RCNN Network"""

    base_config = get_base_config(__file__)

    def __init__(self, config, debug=False, with_rcnn=True, num_classes=None,
                 name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)

        import ipdb; ipdb.set_trace()
        self._cfg = config
        self._num_classes = num_classes
        self._with_rcnn = with_rcnn
        self._debug = debug

        self._anchor_base_size = self._cfg.ANCHOR_BASE_SIZE
        self._anchor_scales = np.array(self._cfg.ANCHOR_SCALES)
        self._anchor_ratios = np.array(self._cfg.ANCHOR_RATIOS)
        # TODO: Value depends on use of VGG vs Resnet (?)
        self._anchor_stride = self._cfg.ANCHOR_STRIDE

        self._anchor_reference = generate_anchors_ref(
            self._anchor_base_size, self._anchor_ratios, self._anchor_scales
        )
        self._num_anchors = self._anchor_reference.shape[0]

        self._rpn_cls_loss_weight = 1.0
        self._rpn_reg_loss_weight = 2.0

        self._rcnn_cls_loss_weight = 1.0
        self._rcnn_reg_loss_weight = 2.0
        self._losses_collections = ['fastercnn_losses']

    def _build(self, image, pretrained_feature_map, gt_boxes, is_training=True):
        """
        Returns bounding boxes and classification probabilities.

        Args:
            image: A tensor with the image.
                Its shape should be `(1, height, width, 3)`.
            gt_boxes: A tensor with all the ground truth boxes of that image.
                Its shape should be `(num_gt_boxes, 4)`
                Where for each gt box we have (x1, y1, x2, y2), in that order.

        Returns:
            classification_prob: A tensor with the softmax probability for
                each of the bounding boxes found in the image.
                Its shape should be: (num_bboxes, num_categories + 1)
            classification_bbox: A tensor with the bounding boxes found.
                It's shape should be: (num_bboxes, 4). For each of the bboxes
                we have (x1, y1, x2, y2)
        """

        self._rpn = RPN(self._num_anchors, debug=self._debug)
        if self._with_rcnn:
            self._rcnn = RCNN(self._num_classes, debug=self._debug)

        image_shape = tf.shape(image)[1:3]

        variable_summaries(
            pretrained_feature_map, 'pretrained_feature_map', ['rpn'])

        all_anchors = self._generate_anchors(pretrained_feature_map)
        rpn_prediction = self._rpn(
            pretrained_feature_map, gt_boxes, image_shape, all_anchors,
            is_training=is_training
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

            # TODO: Missing mapping classification_bbox to real coordinates.
            # (and trimming, and NMS?)
            classification_pred = self._rcnn(
                pretrained_feature_map, rpn_prediction['proposals'], gt_boxes,
                image_shape
            )

            prediction_dict['classification_prediction'] = classification_pred

        if is_training and self._debug:
            with tf.name_scope('draw_bboxes'):
                tf.summary.image('image', image, max_outputs=20)
                tf.summary.image(
                    'top_1_rpn_boxes',
                    draw_bboxes(image, rpn_prediction['proposals'], 1), max_outputs=20
                )
                tf.summary.image(
                    'top_10_rpn_boxes',
                    draw_bboxes(image, rpn_prediction['proposals'], 10),
                    max_outputs=20
                )
                tf.summary.image(
                    'top_20_rpn_boxes',
                    draw_bboxes(image, rpn_prediction['proposals'], 20),
                    max_outputs=20
                )

        return prediction_dict

    def loss(self, prediction_dict):
        """
        Compute the joint training loss for Faster RCNN.
        """

        with tf.name_scope('losses'):
            rpn_loss_dict = self._rpn.loss(
                prediction_dict['rpn_prediction']
            )

            # Losses have a weight assigned.
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
                tf.losses.add_loss(loss_tensor)

            regularization_loss = tf.losses.get_regularization_loss()
            no_reg_loss = tf.losses.get_total_loss(add_regularization_losses=False)
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

            return total_loss

    def _generate_anchors(self, feature_map):
        with tf.variable_scope('generate_anchors'):
            feature_map_shape = tf.shape(feature_map)[1:3]
            grid_width = feature_map_shape[1]
            grid_height = feature_map_shape[0]
            shift_x = tf.range(grid_width) * self._anchor_stride
            shift_y = tf.range(grid_height) * self._anchor_stride
            shift_x, shift_y = meshgrid(shift_x, shift_y)

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

    def load_weights(self, checkpoint_file):
        return self._pretrained.load_weights(checkpoint_file)

    @property
    def summary(self):
        """
        Generate merged summary of all the sub-summaries used inside the
        Faster R-CNN network.
        """
        summaries = [
            tf.summary.merge_all(key=self._losses_collections[0]),
            tf.summary.merge_all(key='rpn'),
        ]

        if self._with_rcnn:
            summaries.append(tf.summary.merge_all(key='rcnn'))

        return tf.summary.merge(summaries)
