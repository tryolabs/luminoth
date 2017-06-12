import sonnet as snt
import tensorflow as tf
import numpy as np

class Dataset(snt.AbstractModule):

    NUM_CLASSES = 20  # default to NUM_CLASSES of PASCAL VOC
    DATASET_DIR = 'datasets/voc/tf'

    def __init__(self, *args, **kwargs):
        self._num_classes = kwargs.pop('num_classes', self.NUM_CLASSES)
        self._dataset_dir = kwargs.pop('dataset_dir', self.DATASET_DIR)
        self._num_epochs = kwargs.pop('num_epochs', 10)
        self._batch_size = kwargs.pop('batch_size', 1)
        self._subset = kwargs.pop('subset', 'train')
        self._losses = {}

        super(Dataset, self).__init__(*args, **kwargs)

    def rpn_cost(self, rpn_cls_prob, rpn_labels, rpn_bbox_target, rpn_bbox_pred):
        """
        Returns cost for Region Proposal Network based on:

        Args:
            rpn_cls_prob: Probability of for being an object for each anchor
                in the image. Shape -> (1, height, width, 2)
            rpn_labels: Ground truth labeling for each anchor. Should be
                1: for positive labels
                0: for negative labels
                -1: for labels we should ignore.
                Shape -> (1, height, width, 4)
            rpn_bbox_target: Bounding box output target for rpn.
            rpn_bbox_pred: Bounding box output prediction for rpn.

        Returns:
            Multiloss between cls probability and bbox target.
        """

        # First, we need to calculate classification loss over `rpn_cls_prob`
        # and `rpn_labels`. Ignoring all anchors where `rpn_labels = -1`.

        # For classification loss we use log loss of two classes. So we need to:
        # - filter `rpn_cls_prob` that are ignored. We need to reshape both labels and prob
        # - transform positive and negative `rpn_labels` to same shape as `rpn_cls_prob`.
        # - then we can use `tf.losses.log_loss` which returns a tensor.


        from IPython.core.debugger import Pdb
        Pdb().set_trace()

        # Flatten labels.
        rpn_labels = tf.cast(tf.reshape(rpn_labels, [-1]), tf.int32)
        # Transform to boolean tensor with True only when != -1 (else == -1 -> False)
        labels_not_ignored = tf.not_equal(rpn_labels, -1)

        # Flatten rpn_cls_prob (only anchors, not completely).
        rpn_cls_prob = tf.reshape(rpn_cls_prob, [-1, 2])

        # Now we only have the labels we are going to compare with the
        # cls probability.
        labels = tf.boolean_mask(rpn_labels, labels_not_ignored)
        cls_prob = tf.boolean_mask(rpn_cls_prob, labels_not_ignored)

        # We need to transform `labels` to `cls_prob` shape.
        labels_prob = tf.one_hot(labels, 2)

        # TODO: In other implementations they use `sparse_softmax_cross_entropy_with_logits` with `reduce_mean`. Should we use that?
        log_loss = tf.losses.log_loss(labels_prob, cls_prob)

        self._losses['rpn_classification_loss'] = log_loss

        # Finally, we need to calculate the regression loss over `rpn_bbox_target`
        # and `rpn_bbox_pred`.
        # Since `rpn_bbox_target` is obtained from AnchorTargetLayer then we
        # just need to apply SmoothL1Loss.
        rpn_bbox_target = tf.reshape(rpn_bbox_target, [-1, 4])
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

        # We only care for positive labels
        positive_labels = tf.equal(rpn_labels, 1)
        rpn_bbox_target = tf.boolean_mask(rpn_bbox_target, positive_labels)
        rpn_bbox_pred = tf.boolean_mask(rpn_bbox_pred, positive_labels)

        smooth_l1_loss = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_target)
        self._losses['rpn_regression_loss'] = smooth_l1_loss

        # TODO: Return loss with correct dimensions
        return log_loss + smooth_l1_loss


    def cost(self, rpn_bbox_prediction, rpn_bbox_target):
        """
        Returns cost for general object detection dataset.

        TODO: Maybe we need many methods, one for each

        Args:
            TODO:

        Returns:
            Multi-loss?
        """
        pass

    def _smooth_l1_loss(self, bbox_prediction, bbox_target, sigma=1.0):
        """
        Smooth L1 loss is defined as:

        0.5 * x^2                  if |x| < d
        abs(x) - 0.5               if |x| >= d

        Where d = 1 and x = prediction - target


        TODO: Implementation copied from TFFRCNN.

        """
        sigma2 = sigma ** 2
        deltas = bbox_prediction - bbox_target
        deltas_abs = tf.abs(deltas)
        smooth_l1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        return tf.reduce_sum(
            tf.square(deltas) * 0.5 * sigma2 * smooth_l1_sign +
            (deltas_abs - 0.5 / sigma2) * tf.abs(smooth_l1_sign - 1),
            [1]
        )
