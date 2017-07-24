import sonnet as snt
import tensorflow as tf

from .roi_pool import ROIPoolingLayer
from .rcnn_target import RCNNTarget
from .rcnn_proposal import RCNNProposal
from .utils.losses import smooth_l1_loss
from .utils.vars import variable_summaries


class RCNN(snt.AbstractModule):
    """RCNN """

    def __init__(self, num_classes, layer_sizes=[4096, 4096], debug=False,
                 name='rcnn'):
        super(RCNN, self).__init__(name=name)
        self._num_classes = num_classes
        self._layer_sizes = layer_sizes
        self._activation = tf.nn.relu6
        self._dropout_keep_prob = 1.

        self._debug = debug

    def _instantiate_layers(self):
        fc_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=1., uniform=True, mode='FAN_AVG'
        )

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)

        self._layers = [
            snt.Linear(
                layer_size,
                name="fc_{}".format(i),
                initializers={'w': fc_initializer},
                regularizers={'w': regularizer},
            )
            for i, layer_size in enumerate(self._layer_sizes)
        ]

        self._classifier_layer = snt.Linear(
            self._num_classes + 1, name="fc_classifier",
            initializers={'w': fc_initializer},
            regularizers={'w': regularizer},
        )

        # TODO: Not random initializer
        self._bbox_layer = snt.Linear(
            self._num_classes * 4, name="fc_bbox",
            initializers={'w': fc_initializer},
            regularizers={'w': regularizer}
        )

        self._roi_pool = ROIPoolingLayer(debug=self._debug)
        self._rcnn_target = RCNNTarget(self._num_classes, debug=self._debug)
        self._rcnn_proposal = RCNNProposal(self._num_classes)

    def _build(self, pretrained_feature_map, proposals, gt_boxes, im_shape):
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
        self._instantiate_layers()

        prediction_dict = {}

        proposals_target, bbox_target = self._rcnn_target(
            proposals, gt_boxes)

        # We flatten to set shape, but it is already a flat Tensor.
        in_batch_proposals = tf.reshape(
            tf.not_equal(proposals_target, -1), [-1]
        )
        roi_proposals = tf.boolean_mask(proposals, in_batch_proposals)
        roi_bbox_target = tf.boolean_mask(bbox_target, in_batch_proposals)
        roi_proposals_target = tf.boolean_mask(proposals_target, in_batch_proposals)

        roi_prediction = self._roi_pool(
            roi_proposals, pretrained_feature_map,
            im_shape
        )

        if self._debug:
            prediction_dict['roi_prediction'] = roi_prediction

        pooled_layer = roi_prediction['roi_pool']

        # We treat num proposals as batch number so that when flattening we
        # get a (num_proposals, flatten_pooled_feature_map_size) Tensor.
        flatten_net = tf.contrib.layers.flatten(pooled_layer)
        net = tf.identity(flatten_net)

        if self._debug:
            prediction_dict['flatten_net'] = net  # TODO: debug tmp
            variable_summaries(pooled_layer, 'pooled_layer', ['rcnn'])

        # After flattening we are lef with a
        # (num_proposals, pool_height * pool_width * 512) tensor.
        # The first dimension works as batch size when applied to snt.Linear.
        for i, layer in enumerate(self._layers):
            net = layer(net)
            prediction_dict['layer_{}_out'.format(i)] = net  # TODO: debug tmp
            net = self._activation(net)
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        cls_score = self._classifier_layer(net)
        prob = tf.nn.softmax(cls_score, dim=1)
        bbox_offsets = self._bbox_layer(net)

        proposal_prediction = self._rcnn_proposal(
            roi_proposals, bbox_offsets, prob, im_shape)

        variable_summaries(prob, 'prob', ['rcnn'])
        variable_summaries(bbox_offsets, 'bbox_offsets', ['rcnn'])

        prediction_dict['cls_score'] = cls_score
        prediction_dict['cls_prob'] = prob
        prediction_dict['bbox_offsets'] = bbox_offsets
        prediction_dict['cls_target'] = roi_proposals_target
        prediction_dict['roi_proposals'] = roi_proposals
        prediction_dict['bbox_offsets_target'] = roi_bbox_target
        prediction_dict['objects'] = proposal_prediction['objects']
        prediction_dict['objects_labels'] = proposal_prediction['proposal_label']
        prediction_dict['objects_labels_prob'] = proposal_prediction['proposal_label_prob']

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
                    cls_target_labeled, depth=self._num_classes + 1,
                    name='cls_target_one_hot'
                )

                # We get cross entropy loss of each proposal.
                cross_entropy_per_proposal = tf.nn.softmax_cross_entropy_with_logits(
                    labels=cls_target_one_hot, logits=cls_score_labeled
                )

                if self._debug:
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

                reg_loss_per_proposal = smooth_l1_loss(
                    bbox_offset_cleaned, bbox_offsets_target_labeled)

                tf.summary.scalar(
                    'rcnn_foreground_samples',
                    tf.shape(bbox_offset_cleaned)[0], ['rcnn']
                )

                if self._debug:
                    prediction_dict['reg_loss_per_proposal'] = reg_loss_per_proposal

                return {
                    'rcnn_cls_loss': tf.reduce_mean(
                        cross_entropy_per_proposal
                    ),
                    'rcnn_reg_loss': tf.reduce_mean(reg_loss_per_proposal),
                }
