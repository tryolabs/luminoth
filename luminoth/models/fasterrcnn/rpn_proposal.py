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
        # After applying NMS we filter the top M anchors.
        # It's important to understand that because of NMS, it is not certain
        # we will have this many output proposals. This is just the upper
        # bound.
        self._post_nms_top_n = config.post_nms_top_n
        # Threshold to use for NMS.
        self._nms_threshold = config.nms_threshold
        # Currently we do not filter out proposals by size.
        self._min_size = config.min_size
        self._filter_outside_anchors = config.filter_outside_anchors
        self._clip_after_nms = config.clip_after_nms
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
                nms_proposals: A Tensor with the final selected proposed
                    bounding boxes. Its shape should be
                    (total_nms_proposals, 4).
                nms_proposals_scores: A Tensor with the probability of being an
                    object for that proposal. Its shape should be
                    (total_nms_proposals, 1)
                proposals: A Tensor with all the valid RPN proposals.
                scores: A Tensor with a score for each of the valid RPN
                    proposals.
                proposals_raw: A Tensor with all the RPN proposals without any
                    filtering.
                scores_raw: A Tensor with a score for each of the unfiltered
                    RPN proposals.
        """
        # Scores are extracted from the second scalar of the cls probability.
        # cls_probability is a softmax of (background, foreground).
        scores = rpn_cls_prob[:, 1]
        # Force flatten the scores (it should be already be flatten).
        scores = tf.reshape(scores, [-1])

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
                scores = tf.boolean_mask(scores, anchor_filter)

        # Decode boxes
        all_proposals = decode(all_anchors, rpn_bbox_pred)

        # Filter proposals with negative or zero area.
        (x_min, y_min, x_max, y_max) = tf.unstack(
            all_proposals, axis=1
        )
        proposal_filter = tf.greater(
            tf.maximum(x_max - x_min, 0.0) * tf.maximum(y_max - y_min, 0.0),
            0.0
        )
        proposal_filter = tf.reshape(proposal_filter, [-1])

        # Filter proposals and scores.
        total_proposals = tf.shape(scores)[0]
        scores = tf.boolean_mask(
            scores, proposal_filter,
            name='filter_invalid_scores'
        )
        proposals = tf.boolean_mask(
            all_proposals, proposal_filter,
            name='filter_invalid_proposals'
        )
        if self._debug:
            proposals_raw = tf.identity(proposals)
            scores_raw = tf.identity(scores)

        if not self._clip_after_nms:
            # Clip proposals to the image.
            proposals = clip_boxes(proposals, im_shape)

        filtered_proposals = tf.shape(scores)[0]

        tf.summary.scalar(
            'valid_proposals_ratio',
            (
                tf.cast(filtered_proposals, tf.float32) /
                tf.cast(total_proposals, tf.float32)
            ), ['rpn'])

        tf.summary.scalar(
            'invalid_proposals', total_proposals - filtered_proposals, ['rpn'])

        # Get top `pre_nms_top_n` indices by sorting the proposals by score.
        k = tf.minimum(self._pre_nms_top_n, tf.shape(scores)[0])
        top_k = tf.nn.top_k(scores, k=k)
        top_k_scores = top_k.values

        top_k_proposals = tf.gather(proposals, top_k.indices)
        # We reorder the proposals into TensorFlows bounding box order for
        # `tf.image.non_max_supression` compatibility.
        proposals_tf_order = change_order(top_k_proposals)

        # We cut the pre_nms filter in pure TF version and go straight into
        # NMS.
        selected_indices = tf.image.non_max_suppression(
            proposals_tf_order, tf.squeeze(top_k_scores), self._post_nms_top_n,
            iou_threshold=self._nms_threshold
        )

        # Selected_indices is a smaller tensor, we need to extract the
        # proposals and scores using it.
        nms_proposals = tf.gather(
            proposals_tf_order, selected_indices, name='gather_nms_proposals'
        )
        nms_proposals_scores = tf.gather(
            top_k_scores, selected_indices, name='gather_nms_proposals_scores'
        )

        # We switch back again to the regular bbox encoding.
        nms_proposals = change_order(nms_proposals)

        if self._clip_after_nms:
            # Clip proposals to the image after NMS.
            nms_proposals = clip_boxes(nms_proposals, im_shape)

        # Adds batch number for consistency and multi image batch support.
        batch_inds = tf.zeros(
            (tf.shape(nms_proposals)[0], 1), dtype=tf.float32
        )
        nms_proposals = tf.concat([batch_inds, nms_proposals], axis=1)

        pred = {
            'nms_proposals': tf.stop_gradient(nms_proposals),
            'nms_proposals_scores': tf.stop_gradient(nms_proposals_scores),
        }

        if self._debug:
            pred.update({
                'proposals': proposals,
                'scores': scores,
                'proposals_raw': proposals_raw,
                'scores_raw': scores_raw,
                'top_k_proposals': top_k_proposals,
                'top_k_scores': top_k_scores,
                'all_proposals': all_proposals
            })

        return pred
