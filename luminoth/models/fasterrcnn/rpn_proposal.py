import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_transform_tf import bbox_decode


class RPNProposal(snt.AbstractModule):
    """Transforms anchors and RPN predictions into object proposals.

    Using the fixed anchors and the RPN predictions for both classification
    and regression (adjusting the bounding box), we return a list of objects
    sorted by relevance.

    Besides applying the transformations (or adjustments) from the prediction,
    it also tries to get rid of duplicate proposals by using non maximum
    supression (NMS).
    """
    def __init__(self, num_anchors, config, name='proposal_layer'):
        super(RPNProposal, self).__init__(name=name)
        self._num_anchors = num_anchors

        # Filtering config
        # Before applying NMS we filter the top N anchors.
        self._pre_nms_top_n = config.pre_nms_top_n
        # After applying NMS we filter the top M anchors.
        # It's important to understand that because of NMS, it is not certain
        # we will have this many output proposals. This is just the upper bound.
        self._post_nms_top_n = config.post_nms_top_n
        # Threshold to use for NMS.
        self._nms_threshold = config.nms_threshold
        # TODO: Currently we do not filter out proposals by size.
        self._min_size = config.min_size

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
                proposals: A Tensor with all the RPN proposals without any
                    filtering.
                scores: A Tensor with a score for each of the unfiltered RPN
                    proposals.
        """
        # Scores are extracted from the second scalar of the cls probability.
        # cls_probability is a softmax of (background, foreground).
        scores = tf.slice(rpn_cls_prob, [0, 1], [-1, -1])
        # Force flatten the scores (it should be already be flatten).
        scores = tf.reshape(scores, [-1])

        # We first, remove anchor that partially fall ouside an image.
        height = im_shape[0]
        width = im_shape[1]

        (x_min_anchor, y_min_anchor,
         x_max_anchor, y_max_anchor) = tf.unstack(all_anchors, axis=1)

        # Filter anchors that are partially outside the image.
        anchor_filter = tf.logical_and(
            tf.logical_and(
                tf.greater_equal(x_min_anchor, 0),
                tf.greater_equal(y_min_anchor, 0)
            ),
            tf.logical_and(
                tf.less(x_max_anchor, width),
                tf.less(y_max_anchor, height)
            )
        )

        # We (force) reshape the filter so that we can use it as a boolean mask
        anchor_filter = tf.reshape(anchor_filter, [-1])

        # Filter anchors, predictions and scores.
        all_anchors = tf.boolean_mask(
            all_anchors, anchor_filter, name='filter_anchors')
        scores = tf.boolean_mask(scores, anchor_filter, name='filter_scores')
        rpn_bbox_pred = tf.boolean_mask(
            tf.cast(rpn_bbox_pred, tf.float32), anchor_filter,
            name='filter_rpn_bbox_pred'
        )

        # Decode boxes
        proposals = bbox_decode(all_anchors, rpn_bbox_pred)

        x_min, y_min, x_max, y_max = tf.unstack(proposals, axis=1)

        # Clip boxes
        image_shape = tf.cast(im_shape, tf.float32)
        x_min = tf.maximum(tf.minimum(x_min, image_shape[1] - 1), 0.)
        y_min = tf.maximum(tf.minimum(y_min, image_shape[0] - 1), 0.)
        x_max = tf.maximum(tf.minimum(x_max, image_shape[1] - 1), 0.)
        y_max = tf.maximum(tf.minimum(y_max, image_shape[0] - 1), 0.)

        proposals = tf.stack([x_min, y_min, x_max, y_max], axis=1)

        # Filter proposals with negative area.
        # TODO: Optional, is not done in paper, maybe we should make it
        # configurable.
        proposal_filter = tf.greater_equal(
            (x_max - x_min) * (y_max - y_min), 0)
        proposal_filter = tf.reshape(proposal_filter, [-1])

        # Filter proposals and scores.
        scores = tf.boolean_mask(
            scores, proposal_filter, name='filter_invalid_scores')
        proposals = tf.boolean_mask(
            proposals, proposal_filter, name='filter_invalid_proposals')

        # Get top `pre_nms_top_n` indices by sorting the proposals by score.
        k = tf.minimum(self._pre_nms_top_n, tf.shape(scores)[0])
        top_k = tf.nn.top_k(scores, k=k)
        scores = top_k.values

        x_min, y_min, x_max, y_max = tf.unstack(proposals, axis=1)
        x_min = tf.gather(x_min, top_k.indices, name='gather_topk_x_min')
        y_min = tf.gather(y_min, top_k.indices, name='gather_topk_y_min')
        x_max = tf.gather(x_max, top_k.indices, name='gather_topk_x_max')
        y_max = tf.gather(y_max, top_k.indices, name='gather_topk_x_max')

        # We reorder the proposals into TensorFlows bounding box order for
        # `tf.image.non_max_supression` compatibility.
        proposals_tf_order = tf.stack([y_min, x_min, y_max, x_max], axis=1)

        # We cut the pre_nms filter in pure TF version and go straight into NMS.
        selected_indices = tf.image.non_max_suppression(
            proposals_tf_order, tf.squeeze(scores), self._post_nms_top_n,
            iou_threshold=self._nms_threshold
        )

        # Selected_indices is a smaller tensor, we need to extract the
        # proposals and scores using it.
        nms_proposals = tf.gather(proposals_tf_order, selected_indices, name='gather_nms_proposals')
        nms_proposals_scores = tf.gather(scores, selected_indices, name='gather_nms_proposals_scores')

        # We switch back again to the regular bbox encoding.
        y_min, x_min, y_max, x_max = tf.split(value=nms_proposals, num_or_size_splits=4, axis=1)
        # Adds batch number for consistency and future multi image batche support.
        batch_inds = tf.zeros((tf.shape(nms_proposals)[0], 1), dtype=tf.float32)
        nms_proposals = tf.concat([batch_inds, x_min, y_min, x_max, y_max], axis=1)

        return {
            'nms_proposals': nms_proposals,
            'nms_proposals_scores': nms_proposals_scores,
            'proposals': proposals,
            'scores': scores,
        }
