import sonnet as snt
import tensorflow as tf
import numpy as np

from .rcnn_target import RCNNTarget
from .rcnn_proposal import RCNNProposal
from .utils.losses import smooth_l1_loss
from .utils.vars import variable_summaries


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
            self._rcnn_proposal = RCNNProposal(self._num_classes)

    def _build(self, pooled_layer, proposals, gt_boxes, im_shape):
        """
        Classifies proposals based on the pooled feature map.

        pooled_layer:
            Feature map
            Shape (num_proposals, pool_height, pool_width, 512).
        proposals:
            Shape (num_proposals, 4)

        TODO: El pooled layer es el volumen con todos los ROI o es uno por cada ROI?
        TODO: Donde puedo comparar los resultados con las labels posta?
        """

        prediction_dict = {}

        # We treat num proposals as batch number so that when flattening we actually
        # get a (num_proposals, flatten_pooled_feature_map_size) Tensor.
        flatten_net = tf.contrib.layers.flatten(pooled_layer)
        net = tf.identity(flatten_net)

        prediction_dict['flatten_net'] = net  # TODO: debug tmp
        # After flattening we are lef with a (num_proposals, pool_height * pool_width * 512) tensor.
        # The first dimension works as batch size when applied to snt.Linear.
        for i, layer in enumerate(self._layers):
            net = layer(net)
            prediction_dict['layer_{}_out'.format(i)] = net  # TODO: debug tmp
            net = self._activation(net)
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)
            variable_summaries(layer.w, 'layer_{}_W'.format(i), ['RCNN'])

        cls_score = self._classifier_layer(net)
        prob = tf.nn.softmax(cls_score, dim=1)
        bbox_offsets = self._bbox_layer(net)

        proposals_target, bbox_target = self._rcnn_target(
            proposals, gt_boxes)

        objects, objects_labels, objects_labels_prob = self._rcnn_proposal(
            proposals, bbox_offsets, prob)

        variable_summaries(prob, 'prob', ['RCNN'])
        variable_summaries(bbox_offsets, 'bbox_offsets', ['RCNN'])

        prediction_dict['cls_score'] = cls_score
        prediction_dict['cls_prob'] = prob
        prediction_dict['bbox_offsets'] = bbox_offsets
        prediction_dict['cls_target'] = proposals_target
        prediction_dict['bbox_offsets_target'] = bbox_target
        prediction_dict['objects'] = objects
        prediction_dict['objects_labels'] = objects_labels
        prediction_dict['objects_labels_prob'] = objects_labels_prob

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
                    We have to compare only the proposals labeled with the offsets
                    for that label.

                bbox_offsets_target: shape (num_proposals, 4)
                    Has the true offset of each proposal for the true label.
                    In case of not having a true label (non-background) then
                    it's just zeroes.

        """
        with self._enter_variable_scope():
            with tf.name_scope('RCNNLoss'):
                cls_score = prediction_dict['cls_score']
                cls_prob = prediction_dict['cls_prob']
                cls_target = tf.cast(prediction_dict['cls_target'], tf.int32)

                # First we need to calculate the log loss betweetn cls_prob and
                # cls_target

                # We only care for the targets that are >= 0
                not_ignored = tf.reshape(tf.greater_equal(
                    cls_target, 0), [-1], name='not_ignored')
                # We apply boolean mask to score, prob and target.
                cls_score_labeled = tf.boolean_mask(
                    cls_score, not_ignored, name='cls_score_labeled')
                cls_prob_labeled = tf.boolean_mask(
                    cls_prob, not_ignored, name='cls_prob_labeled')
                cls_target_labeled = tf.boolean_mask(
                    cls_target, not_ignored, name='cls_target_labeled')

                # Transform to one-hot vector
                cls_target_one_hot = tf.one_hot(
                    cls_target_labeled, depth=self._num_classes + 1, name='cls_target_one_hot')

                # We get cross entropy loss of each proposal.
                cross_entropy_per_proposal = tf.nn.softmax_cross_entropy_with_logits(
                    labels=cls_target_one_hot, logits=cls_score_labeled
                )

                prediction_dict['cross_entropy_per_proposal'] = cross_entropy_per_proposal

                # Second we need to calculate the smooth l1 loss between
                # `bbox_offsets` and `bbox_offsets_target`.
                bbox_offsets = prediction_dict['bbox_offsets']
                bbox_offsets_target = prediction_dict['bbox_offsets_target']

                # We only want the non-background labels bounding boxes.
                not_ignored = tf.reshape(tf.greater(cls_target, 0), [-1])
                bbox_offsets_labeled = tf.boolean_mask(
                    bbox_offsets, not_ignored, name='bbox_offsets_labeled')
                bbox_offsets_target_labeled = tf.boolean_mask(
                    bbox_offsets_target, not_ignored, name='bbox_offsets_target_labeled')

                cls_target_labeled = tf.boolean_mask(
                    cls_target, not_ignored, name='cls_target_labeled')
                # `cls_target_labeled` is based on `cls_target` which has `num_classes` + 1 classes.
                # for making `one_hot` with depth `num_classes` to work we need
                # to lower them to make them 0-index.
                cls_target_labeled = cls_target_labeled - 1

                cls_target_one_hot = tf.one_hot(
                    cls_target_labeled, depth=self._num_classes, name='cls_target_one_hot')

                # cls_target now is (num_labeled, num_classes)
                bbox_flatten = tf.reshape(
                    bbox_offsets_labeled, [-1, 4], name='bbox_flatten')

                # We use the flatten cls_target_one_hot as boolean mask for the
                # bboxes.
                cls_flatten = tf.cast(tf.reshape(
                    cls_target_one_hot, [-1]), tf.bool, 'cls_flatten_as_bool')

                bbox_offset_cleaned = tf.boolean_mask(
                    bbox_flatten, cls_flatten, 'bbox_offset_cleaned')

                reg_loss_per_proposal = smooth_l1_loss(
                    bbox_offset_cleaned, bbox_offsets_target_labeled)

                prediction_dict['reg_loss_per_proposal'] = reg_loss_per_proposal

                # Hack to avoid having nan loss.
                # reg_loss = tf.cond(tf.is_nan(reg_loss), lambda: tf.constant(0.0, dtype=tf.float32), lambda: reg_loss)

                return {
                    'rcnn_cls_loss': tf.reduce_mean(cross_entropy_per_proposal),
                    'rcnn_reg_loss': tf.reduce_mean(reg_loss_per_proposal),
                }
