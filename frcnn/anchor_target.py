import sonnet as snt
import tensorflow as tf
import numpy as np

from .utils.bbox import bbox_overlaps
from .utils.bbox_transform import bbox_transform, unmap


class AnchorTarget(snt.AbstractModule):
    """
    AnchorTarget

    TODO: (copied) Assign anchors to ground-truth targets. Produces anchor
    classification labels and bounding-box regression targets.

    Detailed responsabilities:
    - Keep anchors that are inside the image.
    - We need to set each anchor with a label:
        1 is positive
            when GT overlap is >= 0.7 or for GT max overlap (one anchor)
        0 is negative
            when GT overlap is < 0.3
        -1 is don't care
            useful for subsampling negative labels

    - Create BBox targets with anchors and GT.
        TODO:
        Cual es la diferencia entre bbox_inside_weights y bbox_outside_weights


    Things to take into account:
    - We can assign "don't care" labels to anchors we want to ignore in batch.


    Returns:
        labels: label for each anchor
        bbox_targets: bbox regresion values for each anchor
        bbox_inside_weights: TODO: ??
        bbox_outside_weights: TODO: ??

    """
    def __init__(self, num_anchors, feat_stride=[16], name='anchor_target'):
        super(AnchorTarget, self).__init__(name=name)
        self._num_anchors = num_anchors
        self._feat_stride = feat_stride

        self._allowed_border = 0
        # We set clobber positive to False to make sure that there is always at
        # least one positive anchor per GT box.
        self._clobber_positives = False
        # We set anchors as positive when the IoU is greater than `positive_overlap`.
        self._positive_overlap = 0.7
        # We set anchors as negative when the IoU is less than `negative_overlap`.
        self._negative_overlap = 0.3
        # Fraction of the batch to be foreground labeled anchors.
        self._foreground_fraction = 0.5
        self._batch_size = 256
        # TODO:
        self._bbox_inside_weights = (1.0, 1.0, 1.0, 1.0)

    def _build(self, pretrained_shape, gt_boxes, im_info, all_anchors):
        """
        Args:
            pretrained_shape:
                Shape of the pretrained feature map, (H, W).
            gt_boxes:
                A Tensor with the groundtruth bounding boxes of the image of
                the batch being processed. It's dimensions should be (num_gt, 5).
                The last dimension is used for the label.
            im_info:
                Shape of original image (height, width) in order to define
                anchor targers in respect with gt_boxes.
            all_anchors:
                A Tensor with all the bounding boxes coords of the anchors.

        We currently use the `anchor_target_layer` based on the code provided
        in the original Caffe implementation by Ross Girshick. Ideally we
        should migrate this code to pure Tensorflow tensor-based graph.

        TODO: Tensorflow limitations for the migration.
        TODO: Performance impact of current use of py_func

        Returns:
            Tuple of the tensors of:
                labels: (1, 0, -1) for each anchor.
                    Shape (num_anchors, 1)
                bbox_targets: 4d bbox targets as specified by paper
                    Shape (num_anchors, 4)
                max_overlaps: Max IoU overlap with ground truth boxes.
                    Shape (num_anchors, 1)
        """

        (
            labels, bbox_targets, max_overlaps
        ) = tf.py_func(
            self._anchor_target_layer_np,
            [pretrained_shape, gt_boxes, im_info, all_anchors],
            [tf.float32, tf.float32, tf.float32]

        )

        # TODO: missing bbox_inside_weights, bbox_outside_weights
        return labels, bbox_targets, max_overlaps


    def _anchor_target_layer(self, pretrained_shape, gt_boxes, im_info, all_anchors):
        """
        Function working with Tensors instead of instances for proper
        computing in the Tensorflow graph.
        """
        raise NotImplemented()


    def _anchor_target_layer_np(self, pretrained_shape, gt_boxes, im_info, all_anchors):

        np.random.seed(0)  # TODO: Remove, just for reproducibility
        """
        Function to be executed with tf.py_func
        """

        height, width = pretrained_shape[1:3]
        # We have "W x H x k" anchors
        total_anchors = all_anchors.shape[0]

        # only keep anchors inside the image
        # TODO: We should do this when anchors are original generated in
        # network or does it fuck with our dimensions.
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # Start by ignoring all anchors by default and then pick the ones
        # we care about for training.
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # intersection over union (IoU) overlap between the anchors and the
        # ground truth boxes.
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))

        # Find the closest gt box for each anchor.
        argmax_overlaps = overlaps.argmax(axis=1)

        # Generate array with the IoU value of the closest GT box for each
        # anchor.
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

        # Get the closest anchor for each gt box.
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        # Get the value of the max IoU for the closest anchor for each gt.
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        # Find all the indices that match (at least one, but could be more).
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not self._clobber_positives:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < self._negative_overlap] = 0

        # foreground label: for each ground-truth, anchor with highest overlap
        # When the armax is many items we use all of them (for consistency).
        labels[gt_argmax_overlaps] = 1

        # foreground label: above threshold Intersection over Union (IoU)
        labels[max_overlaps >= self._positive_overlap] = 1

        if self._clobber_positives:
            # assign background labels last so that negative labels can clobber positives
            labels[max_overlaps < self._negative_overlap] = 0

        # subsample positive labels if we have too many
        num_fg = int(self._foreground_fraction * self._batch_size)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self._batch_size - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        # Returns bbox targets with shape (len(inds_inside), 4)
        bbox_targets = self._compute_targets(
            anchors, gt_boxes[argmax_overlaps, :]).astype(np.float32)

        # We unroll "inside anchors" value for all anchors (for shape compatibility)
        labels = unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        max_overlaps = unmap(max_overlaps, total_anchors, inds_inside, fill=0)

        # TODO: Decide what to do with weights

        return labels, bbox_targets, max_overlaps

    def _compute_targets(self, boxes, groundtruth_boxes):
        """Compute bounding-box regression targets for an image."""

        assert boxes.shape[0] == groundtruth_boxes.shape[0]
        assert boxes.shape[1] == 4
        assert groundtruth_boxes.shape[1] == 5

        return bbox_transform(
            boxes, groundtruth_boxes[:, :4]).astype(np.float32, copy=False)
