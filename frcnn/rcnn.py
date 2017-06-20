import sonnet as snt
import tensorflow as tf
import numpy as np

from .rcnn_target import RCNNTarget
from .utils.losses import smooth_l1_loss

class RCNN(snt.AbstractModule):
    """RCNN """

    def __init__(self, num_classes, layer_sizes=[4096, 4096], name='rcnn'):
        super(RCNN, self).__init__(name=name)
        self._num_classes = num_classes
        self._layer_sizes = layer_sizes
        self._activation = tf.nn.relu
        self._dropout_keep_prob = 0.6
        self._instantiate_layers()

    def _instantiate_layers(self):
        with self._enter_variable_scope():

            self._layers = [
                snt.Linear(
                    layer_size,
                    name="fc_{}".format(i),
                )
                for i, layer_size in enumerate(self._layer_sizes)
            ]

            self._classifier_layer = snt.Linear(
                self._num_classes + 1, name="fc_classifier"
            )

            # TODO: Not random initializer
            self._bbox_layer = snt.Linear(
                self._num_classes * 4, name="fc_bbox"
            )

            self._rcnn_target = RCNNTarget(self._num_classes)



    def _build(self, pooled_layer, proposals, gt_boxes):
        """
        TODO: El pooled layer es el volumen con todos los ROI o es uno por cada ROI?
        TODO: Donde puedo comparar los resultados con las labels posta?
        """
        net = tf.contrib.layers.flatten(pooled_layer)
        for i, layer in enumerate(self._layers):
            net = layer(net)
            net = self._activation(net)
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        cls_score = self._classifier_layer(net)
        prob = tf.nn.softmax(cls_score, dim=1)
        bbox_offsets = self._bbox_layer(net)

        proposals_target, bbox_target = self._rcnn_target(proposals, bbox_offsets, prob, gt_boxes)

        return {
            'cls_score': cls_score,
            'cls_prob': prob,
            'bbox_offsets': bbox_offsets,
            'cls_target': proposals_target,
            'bbox_offsets_target': bbox_target,
        }

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
                    We have to compare only the proposals labeled with the offsets
                    for that label.

                bbox_offsets_target: shape (num_proposals, 4)
                    Has the true offset of each proposal for the true label.
                    In case of not having a true label (non-background) then
                    it's just zeroes.

        """
        cls_score = prediction_dict['cls_score']
        cls_prob = prediction_dict['cls_prob']
        cls_target = tf.cast(prediction_dict['cls_target'], tf.int32)

        # First we need to calculate the log loss betweetn cls_prob and cls_target

        # We only care for the targets that are >= 0
        not_ignored = tf.reshape(tf.greater_equal(cls_target, 0), [-1])
        # We apply boolean mask to both prob and target.
        cls_prob_labeled = tf.boolean_mask(cls_prob, not_ignored)
        cls_target_labeled = tf.boolean_mask(cls_target, not_ignored)

        # Transform to one-hot vector
        cls_target_one_hot = tf.one_hot(
            cls_target_labeled, depth=self._num_classes + 1)

        # TODO: Same doubt as RPN, should we use sparse_softmax_cross_entropy?
        cls_loss = tf.losses.log_loss(cls_target_one_hot, cls_prob_labeled)

        # Second we need to calculate the smooth l1 loss between
        # `bbox_offsets` and `bbox_offsets_target`.
        bbox_offsets = prediction_dict['bbox_offsets']
        bbox_offsets_target = prediction_dict['bbox_offsets_target']

        # We only want the non-background labels bounding boxes.
        not_ignored = tf.reshape(tf.greater(cls_target, 0), [-1])
        bbox_offsets_labeled = tf.boolean_mask(bbox_offsets, not_ignored)
        bbox_offsets_target_labeled = tf.boolean_mask(
            bbox_offsets_target, not_ignored)

        cls_target_labeled = tf.boolean_mask(cls_target, not_ignored)
        cls_target_one_hot = tf.one_hot(
            cls_target_labeled, depth=self._num_classes)

        # cls_target now is (num_labeled, num_classes)
        bbox_flatten = tf.reshape(bbox_offsets_labeled, [-1, 4])

        # We use the flatten cls_target_one_hot as boolean mask for the bboxes.
        cls_flatten = tf.cast(tf.reshape(cls_target_one_hot, [-1]), tf.bool)

        bbox_offset_cleaned = tf.boolean_mask(bbox_flatten, cls_flatten)

        reg_loss = smooth_l1_loss(bbox_offset_cleaned, bbox_offsets_target_labeled)

        return {
            'rcnn_cls_loss': cls_loss,
            'rcnn_reg_loss': reg_loss,
        }
