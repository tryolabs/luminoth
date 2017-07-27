import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_transform_tf import bbox_decode


class RPNProposal(snt.AbstractModule):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").

    Applies NMS and top-N filtering to proposals to limit the number of proposals.

    TODO: Better documentation.
    """
    def __init__(self, num_anchors, name='proposal_layer'):
        super(RPNProposal, self).__init__(name=name)
        self._num_anchors = num_anchors

        # Filtering config  TODO: Use external configuration
        self._pre_nms_top_n = 12000
        self._post_nms_top_n = 2000
        self._nms_threshold = 0.6
        self._min_size = 0  # TF CMU paper suggests removing min size limit -> not used

    def _build(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape):
        scores = tf.slice(rpn_cls_prob, [0, 1], [-1, -1])
        scores = tf.identity(scores, name='scores_after_slice')
        scores = tf.reshape(scores, [-1])
        scores = tf.identity(scores, name='scores_after_reshape')

        # We first, remove anchor that partially fall ouside an image.
        height = im_shape[0]
        width = im_shape[1]

        (x_min_anchor, y_min_anchor,
         x_max_anchor, y_max_anchor) = tf.unstack(all_anchors, axis=1)

        # Filter anchors that are partially outside the image. TODO: Revisar
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

        # We (force) reshape the filter so that we can use it as a boolean mask.
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

        # Filter proposals with negative area. TODO: Optional, is not done in paper, can it happen (given log of width ratio?)
        proposal_filter = tf.greater_equal(
            (x_max - x_min) * (y_max - y_min), 0)
        proposal_filter = tf.reshape(proposal_filter, [-1])

        # Filter proposals and scores.
        scores = tf.boolean_mask(
            scores, proposal_filter, name='filter_invalid_scores')
        proposals = tf.boolean_mask(
            proposals, proposal_filter, name='filter_invalid_proposals')

        # Get pre NMS top selections
        k = tf.minimum(self._pre_nms_top_n, tf.shape(scores)[0])
        top_k = tf.nn.top_k(scores, k=k)
        scores = top_k.values

        x_min, y_min, x_max, y_max = tf.unstack(proposals, axis=1)
        x_min = tf.gather(x_min, top_k.indices, name='gather_topk_x_min')
        y_min = tf.gather(y_min, top_k.indices, name='gather_topk_y_min')
        x_max = tf.gather(x_max, top_k.indices, name='gather_topk_x_max')
        y_max = tf.gather(y_max, top_k.indices, name='gather_topk_x_max')

        # proposals = tf.gather(proposals, top_k.indices, name='gather_topk_proposals')

        # We reorder the proposals for non_max_supression compatibility.

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
