import sonnet as snt
import tensorflow as tf
import numpy as np

from utils.bbox_transform import bbox_transform_inv, clip_boxes
from utils.nms import nms

class ProposalLayer(snt.AbstractModule):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").

    Applies NMS and top-N filtering to proposals to limit the number of proposals.

    TODO: Better documentation.
    """
    def __init__(self, anchors, feat_stride=[16], name='proposal_layer'):
        super(ProposalLayer, self).__init__(name=name)
        self._anchors = anchors
        self._num_anchors = self.anchors.shape[0]
        self._feat_stride = feat_stride

        # Filtering config  TODO: Use external configuration
        self._pre_nms_top_n = 12000
        self._post_nms_top_n = 2000
        self._nms_threshold = 0.7
        self._min_size = 0  # TF CMU paper suggests removing min size limit

    def _build(self, rpn_cls_prob, rpn_bbox_pred):
        rois, rois_scores = tf.py_func(
            self._proposal_layer_np, [rpn_cls_prob, rpn_bbox_pred],
            [tf.float32, tf.float32]
        )

        return rois, rois_scores


    def _proposal_layer_tf(self, rpn_cls_prob, rpn_bbox_pred):
        """
        Function working with Tensors instead of instances for proper
        computing in the Tensorflow graph.
        """
        raise NotImplemented()


    def _proposal_layer_np(self, rpn_cls_prob, rpn_bbox_pred):
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
        """
        scores = rpn_cls_prob[:, :, :, self._num_anchors:]

        # TODO: Why reshape?
        rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
        scores = scores.reshape((-1, 1))

        # 1. Generate proposals from bbox deltas and shifted anchors
        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(self._anchors, rpn_bbox_pred)

        # 2. Clip predicted boxes to image
        im_shape = rpn_cls_prob.shape[1:3]
        proposals = clip_boxes(proposals, im_shape)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_shape[1])
        if self._min_size > 0:
            keep = _filter_boxes(proposals, self._min_size * im_shape[1])
            proposals = proposals[keep, :]
            scores = scores[keep]

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
