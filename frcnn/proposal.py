import sonnet as snt
import tensorflow as tf
import numpy as np

from .utils.bbox_transform import bbox_transform_inv, clip_boxes
from .utils.nms import nms

class Proposal(snt.AbstractModule):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").

    Applies NMS and top-N filtering to proposals to limit the number of proposals.

    TODO: Better documentation.

    TODO: Rename to RPNProposal?
    """
    def __init__(self, num_anchors, feat_stride=[16], name='proposal_layer'):
        super(Proposal, self).__init__(name=name)
        self._num_anchors = num_anchors
        self._feat_stride = feat_stride

        # Filtering config  TODO: Use external configuration
        self._pre_nms_top_n = 12000  # TODO: Not used in TF version
        self._post_nms_top_n = 1000
        self._nms_threshold = 0.7
        self._min_size = 0  # TF CMU paper suggests removing min size limit -> not used

    def _build(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape):
        # rois, rois_scores = tf.py_func(
        #     self._proposal_layer_np, [rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape],
        #     [tf.float32, tf.float32]
        # )

        # TODO: Better verification that proposal_layer_tf is working correctly.
        rois_tf, rois_scores_tf = self._proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape)

        return rois_tf, rois_scores_tf


    def _proposal_layer_tf(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape):
        """
        Function working with Tensors instead of instances for proper
        computing in the Tensorflow graph.
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

    def _proposal_layer_np(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape):
        """
        Function to be executed with tf.py_func

        Comment from original codebase:
            Algorithm:
                for each (H, W) location i
                  generate A anchor boxes centered on cell i
                  apply predicted bbox deltas at cell i to each of the A anchors
                clip predicted boxes to image
                remove predicted boxes with either height or width < threshold
                sort all (proposal, score) pairs by score from highest to lowest
                take top pre_nms_topN proposals before NMS
                apply NMS with threshold 0.7 to remaining proposals
                take after_nms_topN proposals after NMS
                return the top proposals (-> RoIs top, scores top)

        Args:
            rpn_cls_prob:
                Objectness probability for each anchor.
                Shape (batch_size, H, W, num_anchors * 2)
            rpn_bbox_pred:
                RPN bbox delta for each anchor
                Shape (batch_size, H, W, num_anchors * 4)
        """
        # Lets take only the first probability (objectness)
        scores = rpn_cls_prob[:, :, :, self._num_anchors:]

        # Flatten bbox and scores to have a one to one easy matching.
        rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
        scores = scores.reshape((-1, 1))

        # 1. Generate proposals from bbox deltas and shifted anchors
        # Convert anchors into proposals via bbox transformations
        # TODO: Dudas si realmente necesita _anchors o realmente las boxes. PORQUE SOY ESTUPIDO?!
        # TODO: Kill me
        proposals = bbox_transform_inv(all_anchors, rpn_bbox_pred)

        # 2. Clip predicted boxes to image
        proposals = clip_boxes(proposals, im_shape)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        order = scores.ravel().argsort()[::-1]

        # 5. take top pre_nms_topN (e.g. 6000)
        if self._pre_nms_top_n > 0:
            order = order[:self._pre_nms_top_n]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), self._nms_threshold)
        if self._post_nms_top_n > 0:
            keep = keep[:self._post_nms_top_n]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

        return blob, scores
