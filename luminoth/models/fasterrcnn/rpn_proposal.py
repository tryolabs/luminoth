import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_transform_tf import decode, clip_boxes, change_order


class RPNProposal(snt.AbstractModule):
    """Transforms anchors and RPN predictions into object proposals.

    Using the fixed anchors and the RPN predictions for both classification
    and regression (adjusting the bounding box), we return a list of objects
    sorted by relevance.

    Besides applying the transformations (or adjustments) from the prediction,
    it tries to get rid of duplicate proposals by using non maximum supression
    (NMS).
    """
    def __init__(self, num_anchors, config, debug=False,
                 name='proposal_layer'):
        super(RPNProposal, self).__init__(name=name)
        self._num_anchors = num_anchors

        # Filtering config
        # Before applying NMS we filter the top N anchors.
        self._pre_nms_top_n = config.pre_nms_top_n
        self._apply_nms = config.apply_nms
        # After applying NMS we filter the top M anchors.
        # It's important to understand that because of NMS, it is not certain
        # we will have this many output proposals. This is just the upper
        # bound.
        self._post_nms_top_n = config.post_nms_top_n
        # Threshold to use for NMS.
        self._nms_threshold = float(config.nms_threshold)
        # Currently we do not filter out proposals by size.
        self._min_size = config.min_size
        self._filter_outside_anchors = config.filter_outside_anchors
        self._clip_after_nms = config.clip_after_nms
        self._min_prob_threshold = float(config.min_prob_threshold)
        self._debug = debug

    def _build(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape):
        """

        Args:
            rpn_cls_prob: A Tensor with the softmax output for each anchor.
                Its shape should be (total_anchors, 2), with the probability of
                being background and the probability of being foreground for
                each anchor.
            rpn_bbox_pred: A Tensor with the regression output for each anchor.
                Its shape should be (total_anchors, 4).
            all_anchors: A Tensor with the anchors bounding boxes of shape
                (total_anchors, 4), having (x_min, y_min, x_max, y_max) for
                each anchor.
            im_shape: A Tensor with the image shape in format (height, width).

        Returns:
            prediction_dict with the following keys:
                proposals: A Tensor with the final selected proposed
                    bounding boxes. Its shape should be
                    (total_proposals, 4).
                scores: A Tensor with the probability of being an
                    object for that proposal. Its shape should be
                    (total_proposals, 1)
        """
        # Scores are extracted from the second scalar of the cls probability.
        # cls_probability is a softmax of (background, foreground).
        all_scores = rpn_cls_prob[:, 1]
        # Force flatten the scores (it should be already be flatten).
        all_scores = tf.reshape(all_scores, [-1])

        if self._filter_outside_anchors:
            with tf.name_scope('filter_outside_anchors'):
                (x_min_anchor, y_min_anchor,
                 x_max_anchor, y_max_anchor) = tf.unstack(all_anchors, axis=1)

                anchor_filter = tf.logical_and(
                    tf.logical_and(
                        tf.greater_equal(x_min_anchor, 0),
                        tf.greater_equal(y_min_anchor, 0)
                    ),
                    tf.logical_and(
                        tf.less(x_max_anchor, im_shape[1]),
                        tf.less(y_max_anchor, im_shape[0])
                    )
                )
                anchor_filter = tf.reshape(anchor_filter, [-1])
                all_anchors = tf.boolean_mask(
                    all_anchors, anchor_filter, name='filter_anchors')
                rpn_bbox_pred = tf.boolean_mask(rpn_bbox_pred, anchor_filter)
                all_scores = tf.boolean_mask(all_scores, anchor_filter)

        # Decode boxes
        all_proposals = decode(all_anchors, rpn_bbox_pred)

        # Filter proposals with less than threshold probability.
        min_prob_filter = tf.greater_equal(
            all_scores, self._min_prob_threshold
        )

        # Filter proposals with negative or zero area.
        (x_min, y_min, x_max, y_max) = tf.unstack(all_proposals, axis=1)
        zero_area_filter = tf.greater(
            tf.maximum(x_max - x_min, 0.0) * tf.maximum(y_max - y_min, 0.0),
            0.0
        )
        proposal_filter = tf.logical_and(zero_area_filter, min_prob_filter)

        # Filter proposals and scores.
        all_proposals_total = tf.shape(all_scores)[0]
        unsorted_scores = tf.boolean_mask(
            all_scores, proposal_filter,
            name='filtered_scores'
        )
        unsorted_proposals = tf.boolean_mask(
            all_proposals, proposal_filter,
            name='filtered_proposals'
        )
        if self._debug:
            proposals_unclipped = tf.identity(unsorted_proposals)

        if not self._clip_after_nms:
            # Clip proposals to the image.
            unsorted_proposals = clip_boxes(unsorted_proposals, im_shape)

        filtered_proposals_total = tf.shape(unsorted_scores)[0]

        tf.summary.scalar(
            'valid_proposals_ratio',
            (
                tf.cast(filtered_proposals_total, tf.float32) /
                tf.cast(all_proposals_total, tf.float32)
            ), ['rpn'])

        tf.summary.scalar(
            'invalid_proposals',
            all_proposals_total - filtered_proposals_total, ['rpn'])

        # Get top `pre_nms_top_n` indices by sorting the proposals by score.
        k = tf.minimum(self._pre_nms_top_n, tf.shape(unsorted_scores)[0])
        top_k = tf.nn.top_k(unsorted_scores, k=k)

        sorted_top_proposals = tf.gather(unsorted_proposals, top_k.indices)
        sorted_top_scores = top_k.values

        if self._apply_nms:
            with tf.name_scope('nms'):
                # We reorder the proposals into TensorFlows bounding box order
                # for `tf.image.non_max_supression` compatibility.
                proposals_tf_order = change_order(sorted_top_proposals)
                # We cut the pre_nms filter in pure TF version and go straight
                # into NMS.
                selected_indices = tf.image.non_max_suppression(
                    proposals_tf_order, tf.reshape(
                        sorted_top_scores, [-1]
                    ),
                    self._post_nms_top_n, iou_threshold=self._nms_threshold
                )

                # Selected_indices is a smaller tensor, we need to extract the
                # proposals and scores using it.
                nms_proposals_tf_order = tf.gather(
                    proposals_tf_order, selected_indices,
                    name='gather_nms_proposals'
                )

                # We switch back again to the regular bbox encoding.
                proposals = change_order(nms_proposals_tf_order)
                scores = tf.gather(
                    sorted_top_scores, selected_indices,
                    name='gather_nms_proposals_scores'
                )
        else:
            proposals = sorted_top_proposals
            scores = sorted_top_scores

        if self._clip_after_nms:
            # Clip proposals to the image after NMS.
            proposals = clip_boxes(proposals, im_shape)

        pred = {
            'proposals': proposals,
            'scores': scores,
        }

        if self._debug:
            pred.update({
                'sorted_top_scores': sorted_top_scores,
                'sorted_top_proposals': sorted_top_proposals,
                'unsorted_proposals': unsorted_proposals,
                'unsorted_scores': unsorted_scores,
                'all_proposals': all_proposals,
                'all_scores': all_scores,
                # proposals_unclipped has the unsorted_scores scores
                'proposals_unclipped': proposals_unclipped,
            })

        return pred
