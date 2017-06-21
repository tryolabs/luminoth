import numpy as np
import sonnet as snt
import tensorflow as tf

from .dataset import TFRecordDataset
from .pretrained import VGG
from .rcnn import RCNN
from .roi_pool import ROIPoolingLayer
from .rpn import RPN

from .utils.generate_anchors import generate_anchors as generate_anchors_reference
from .utils.ops import meshgrid



def draw_bboxes(image, bboxes, topn=10):
    # change fucking order
    #. we asume bboxes has batch
    bboxes = tf.slice(bboxes, [0, 0], [topn, 5])
    batch, x1, y1, x2, y2 = tf.split(value=bboxes, num_or_size_splits=5, axis=1)

    x1 = x1 / tf.cast(tf.shape(image)[2], tf.float32)
    y1 = y1 / tf.cast(tf.shape(image)[1], tf.float32)
    x2 = x2 / tf.cast(tf.shape(image)[2], tf.float32)
    y2 = y2 / tf.cast(tf.shape(image)[1], tf.float32)

    bboxes = tf.concat([batch, y1, x1, y2, x2], axis=1)
    bboxes = tf.expand_dims(bboxes, 0)
    return tf.image.draw_bounding_boxes(image, bboxes)


class FasterRCNN(snt.AbstractModule):
    """Faster RCNN Network"""

    def __init__(self, config, num_classes=None, name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)

        self._cfg = config
        self._num_classes = num_classes

        # TODO: Better module config
        # if not isinstance(anchor_scales, collections.Iterable):
        #     raise TypeError("anchor_scales must be iterable")
        # anchor_scales = tuple(anchor_scales)

        # if not isinstance(anchor_ratios, collections.Iterable):
        #     raise TypeError("anchor_ratios must be iterable")
        # anchor_ratios = tuple(anchor_ratios)

        # if not isinstance(kernel_shape, collections.Iterable):
        #     raise TypeError("kernel_shape must be iterable")
        # kernel_shape = tuple(kernel_shape)

        # if not anchor_scales:
        #     raise ValueError("anchor_scales must not be empty")
        # if not anchor_ratios:
        #     raise ValueError("anchor_ratios must not be empty")
        # self._anchor_scales = anchor_scales
        # self._anchor_ratios = anchor_ratios

        self._rpn_base_size = 16
        self._rpn_scales = np.array([8, 16, 32])
        self._rpn_ratios = np.array([0.5, 1, 2])
        self._rpn_stride = 16
        self._anchor_reference = generate_anchors_reference(
            self._rpn_base_size, self._rpn_ratios, self._rpn_scales
        )
        self._num_anchors = self._anchor_reference.shape[0]

        with self._enter_variable_scope():
            self._pretrained = VGG()
            self._rpn = RPN(self._num_anchors)
            self._roi_pool = ROIPoolingLayer()
            self._rcnn = RCNN(self._num_classes)

    def _build(self, image, gt_boxes, is_training=True):
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
        image_shape = tf.shape(image)[1:3]
        pretrained_output = self._pretrained(image)
        all_anchors = self._generate_anchors(pretrained_output)
        rpn_prediction = self._rpn(
            pretrained_output, gt_boxes, image_shape, all_anchors, is_training=is_training)

        roi_pool = self._roi_pool(rpn_prediction['proposals'], pretrained_output)

        # TODO: Missing mapping classification_bbox to real coordinates.
        # (and trimming, and NMS?)
        # TODO: Missing gt_boxes labels!
        classification_prediction = self._rcnn(roi_pool, rpn_prediction['proposals'], gt_boxes)

        # We need to apply bbox_transform_inv using classification_bbox delta and NMS?
        drawn_image = draw_bboxes(image, rpn_prediction['proposals'])

        tf.summary.image('image', image, max_outputs=20)
        tf.summary.image('top_10_rpn_boxes', drawn_image, max_outputs=20)
        # TODO: We should return a "prediction_dict" with all the required tensors (for results, loss and monitoring)
        return {
            'all_anchors': all_anchors,
            'rpn_prediction': rpn_prediction,
            'classification_prediction': classification_prediction,
            'roi_pool': roi_pool,
            'gt_boxes': gt_boxes,
            'rpn_drawn_image': drawn_image,
        }

    def load_from_checkpoints(self, checkpoints):
        pass

    def loss(self, prediction_dict):
        """
        Compute the joint training loss for Faster RCNN.
        """

        rpn_loss_dict = self._rpn.loss(
            prediction_dict['rpn_prediction']
        )

        rcnn_loss_dict = self._rcnn.loss(
            prediction_dict['classification_prediction']
        )

        for loss_tensor in list(rpn_loss_dict.values()) + list(rcnn_loss_dict.values()):
            tf.losses.add_loss(loss_tensor)

        # TODO: Should we use get_total_loss here?
        return tf.losses.get_total_loss()

    def _generate_anchors(self, feature_map):

        feature_map_shape = tf.shape(feature_map)[1:3]
        grid_width = feature_map_shape[1]
        grid_height = feature_map_shape[0]
        shift_x = tf.range(grid_width) * self._rpn_stride
        shift_y = tf.range(grid_height) * self._rpn_stride
        shift_x, shift_y = meshgrid(shift_x, shift_y)

        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        shifts = tf.stack(
            [shift_x, shift_y, shift_x, shift_y],
            axis=0
        )

        shifts = tf.transpose(shifts)
        # Shifts now is a (H x W, 4) Tensor

        # TODO: We should implement anchor_reference as Tensor
        num_anchors = self._anchor_reference.shape[0]
        num_anchor_points = tf.shape(shifts)[0]

        all_anchors = (
            self._anchor_reference.reshape((1, num_anchors, 4)) +
            tf.transpose(tf.reshape(shifts, (1, num_anchor_points, 4)), (1, 0, 2))
        )

        all_anchors = tf.reshape(all_anchors, (num_anchors * num_anchor_points, 4))

        return all_anchors
