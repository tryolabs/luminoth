import numpy as np
import sonnet as snt
import tensorflow as tf

from .anchor_target import AnchorTarget
from .config import Config
from .dataset import TFRecordDataset
from .pretrained import VGG
from .proposal import Proposal
from .rcnn import RCNN
from .roi_pool import ROIPoolingLayer
from .rpn import RPN


class FasterRCNN(snt.AbstractModule):
    """Faster RCNN Network"""

    def __init__(self, config, num_classes=None, name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)

        self._cfg = config
        self._num_classes = num_classes

        with self._enter_variable_scope():
            self._pretrained = VGG()
            self._rpn = RPN(self._cfg.ANCHOR_SCALES, self._cfg.ANCHOR_RATIOS)
            self._anchor_target = AnchorTarget(self._rpn.anchors)
            self._proposal = Proposal(self._rpn.anchors)
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
        rpn_layers = self._rpn(pretrained_output, is_training=is_training)
        # TODO: We should rename the
        rpn_cls_prob = rpn_layers['rpn_cls_prob_reshape']
        rpn_bbox_pred = rpn_layers['rpn_bbox_pred']

        rpn_labels, rpn_bbox = self._anchor_target(
            rpn_layers['rpn_cls_score_reshape'], gt_boxes, image_shape)

        blob, scores = self._proposal(
            rpn_layers['rpn_cls_prob'], rpn_layers['rpn_bbox_pred'])
        roi_pool = self._roi_pool(blob, pretrained_output)

        # TODO: Missing mapping classification_bbox to real coordinates.
        # (and trimming, and NMS?)
        classification_prob, classification_bbox = self._rcnn(roi_pool)

        # TODO: We are returning only rpn tensors for training RPN.
        return rpn_cls_prob, rpn_labels, rpn_bbox, rpn_bbox_pred


    def load_from_checkpoints(self, checkpoints):
        pass
