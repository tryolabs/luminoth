import sonnet as snt
import tensorflow as tf
import numpy as np

from .utils.bbox_transform import bbox_transform_inv, clip_boxes
from .utils.nms import nms

class RPNProposal(snt.AbstractModule):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").

    Applies NMS and top-N filtering to proposals to limit the number of proposals.

    TODO: Better documentation.
    """
    def __init__(self, num_anchors, feat_stride=[16], name='proposal_layer'):
        super(RPNProposal, self).__init__(name=name)
        self._num_anchors = num_anchors
        self._feat_stride = feat_stride

        # Filtering config  TODO: Use external configuration
        self._pre_nms_top_n = 12000  # TODO: Not used in TF version
        self._post_nms_top_n = 1000
        self._nms_threshold = 0.7
        self._min_size = 0  # TF CMU paper suggests removing min size limit -> not used

    def _build(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape):
        """
        TODO: Comments (for review you can find an old version in the logs when it was called proposal.py)
        """

        scores = tf.slice(rpn_cls_prob, [0, 0, 0, self._num_anchors], [-1, -1, -1, -1])
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, (-1, 4))
        scores = tf.reshape(scores, (-1, 1))

        # Convert anchors + transformations to proposals
        x_min_anchor, y_min_anchor, x_max_anchor, y_max_anchor = tf.split(value=all_anchors, num_or_size_splits=4, axis=1)
        widths_anchor = tf.cast(x_max_anchor - x_min_anchor, tf.float32)
        heights_anchor = tf.cast(y_max_anchor - y_min_anchor, tf.float32)

        # After getting the width and heights of anchors we get the center X, Y.
        ctr_x = tf.cast(x_min_anchor, tf.float32) + .5 * widths_anchor
        ctr_y = tf.cast(y_min_anchor, tf.float32) + 0.5 * heights_anchor

        # The dx, dy deltas are relative while the dw, dh deltas are "log relative"
        # d[:, x::y] is used for having a `(num_boxes, 1)` shape instead of
        # `(num_boxes,)`
        dx, dy, dw, dh = tf.split(value=rpn_bbox_pred, num_or_size_splits=4, axis=1)

        # We get the center of the real box as center anchor + relative width increaste
        # TODO: why?!
        pred_ctr_x = tf.multiply(dx, widths_anchor) + ctr_x
        pred_ctr_y = tf.multiply(dy, heights_anchor) + ctr_y

        # New width and height using exp
        pred_w = tf.multiply(tf.exp(dw), widths_anchor)
        pred_h = tf.multiply(tf.exp(dh), heights_anchor)

        # x1
        x_min = pred_ctr_x - tf.scalar_mul(0.5, pred_w)
        # y1
        y_min = pred_ctr_y - 0.5 * pred_h
        # x2
        x_max = pred_ctr_x + 0.5 * pred_w
        # y2
        y_max = pred_ctr_y + 0.5 * pred_h

        # Clip boxes
        # x_min, y_min, x_max, y_max = tf.split(value=proposals, num_or_size_splits=4, axis=1)
        x_min_clipped = tf.maximum(tf.minimum(x_min, tf.cast(im_shape[1], tf.float32)), 0.)
        y_min_clipped = tf.maximum(tf.minimum(y_min, tf.cast(im_shape[0], tf.float32)), 0.)
        x_max_clipped = tf.maximum(tf.minimum(x_max, tf.cast(im_shape[1], tf.float32)), 0.)
        y_max_clipped = tf.maximum(tf.minimum(y_max, tf.cast(im_shape[0], tf.float32)), 0.)

        # We reorder the proposals for non_max_supression compatibility
        proposals = tf.concat([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped], axis=1)

        # We cut the pre_nms cut in tf version and go straight into nms
        selected_indices = tf.image.non_max_suppression(proposals, tf.squeeze(scores), self._post_nms_top_n, iou_threshold=self._nms_threshold)
        proposals = tf.gather(proposals, selected_indices)
        scores = tf.gather(scores, selected_indices)

        y_min, x_min, y_max, x_max = tf.split(value=proposals, num_or_size_splits=4, axis=1)
        batch_inds = tf.zeros((tf.shape(proposals)[0], 1), dtype=tf.float32)
        blobs = tf.concat([batch_inds, x_min, y_min, x_max, y_max], axis=1)
        return blobs, scores

        return rois_tf, rois_scores_tf

