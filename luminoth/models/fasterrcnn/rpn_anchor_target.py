import sonnet as snt
import tensorflow as tf
import numpy as np

from luminoth.utils.bbox import bbox_overlaps
from luminoth.utils.bbox_transform import encode, unmap


class RPNAnchorTarget(snt.AbstractModule):
    """RPNAnchorTarget: Get RPN's classification and regression targets.

    RPNAnchorTarget is responsable for calculating the correct values for both
    classification and regression problems. It is also responsable for defining
    which anchors and target values are going to be used for the RPN minibatch.

    For calculating the correct values for classification, being classification
    the question of "does this anchor refer to an object?" returning an
    objectiveness score, we calculate the intersection over union (IoU) between
    the anchors boxes and the ground truth boxes and assign values. When the
    intersection between anchors and groundtruth is above a threshold, we can
    mark the anchor as an object or as being foreground.
    In case of not having any intersection or having a low IoU value, then we
    say that the anchor refers to background.

    For calculating the correct values for the regression, the problem of
    transforming the fixed size anchor into a more suitable bounding box (equal
    to the ground truth box) only applies to some of the anchors, the ones that
    we consider to be foreground.

    RPNAnchorTarget is also responsable for selecting which ones of the anchors
    are going to be used for the minibatch. This is a random process with some
    restrictions on the ratio between foreground and background samples.

    For selecting the minibatch, labels are not only set to 0 or 1, for the
    cases of being background and foreground respectively, but also to -1 for
    the anchors we just want to ignore and not include in the minibatch.

    In summary:
    - 1 is positive
        when GT overlap is >= 0.7 (configurable) or for GT max overlap (one
        anchor)
    - 0 is negative
        when GT overlap is < 0.3 (configurable)
    -1 is don't care
        useful for subsampling negative labels

    Returns:
        labels: label for each anchor
        bbox_targets: bbox regresion values for each anchor
    """
    def __init__(self, num_anchors, config, debug=False, name='anchor_target'):
        super(RPNAnchorTarget, self).__init__(name=name)
        self._num_anchors = num_anchors

        self._allowed_border = config.allowed_border
        # We set clobber positive to False to make sure that there is always at
        # least one positive anchor per GT box.
        self._clobber_positives = config.clobber_positives
        # We set anchors as positive when the IoU is greater than
        # `positive_overlap`.
        self._positive_overlap = config.foreground_threshold
        # We set anchors as negative when the IoU is less than
        # `negative_overlap`.
        self._negative_overlap = config.background_threshold_high
        # Fraction of the batch to be foreground labeled anchors.
        self._foreground_fraction = config.foreground_fraction
        self._minibatch_size = config.minibatch_size

        self._debug = debug
        if self._debug:
            tf.logging.warning(
                'Using RPN Anchor Target in debug mode makes random seed '
                'to be fixed at 0.'
            )

    def _build(self, pretrained_shape, gt_boxes, im_size, all_anchors):
        """
        Args:
            pretrained_shape:
                Shape of the pretrained feature map, (H, W).
            gt_boxes:
                A Tensor with the groundtruth bounding boxes of the image of
                the batch being processed. It's dimensions should be
                (num_gt, 5). The last dimension is used for the label.
            im_size:
                Shape of original image (height, width) in order to define
                anchor targers in respect with gt_boxes.
            all_anchors:
                A Tensor with all the bounding boxes coords of the anchors.

        We currently use the `anchor_target_layer` based on the code provided
        in the original Caffe implementation by Ross Girshick. Ideally we
        should migrate this code to pure Tensorflow tensor-based graph.

        TODO: Tensorflow limitations for the migration.
            Using random.
        TODO: Performance impact of current use of py_func.

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
            [pretrained_shape, gt_boxes, im_size, all_anchors],
            [tf.float32, tf.float32, tf.float32],
            stateful=False,
            name='anchor_target_layer_np'

        )

        return labels, bbox_targets, max_overlaps

    def _anchor_target_layer(self, pretrained_shape, gt_boxes, im_size, all_anchors):
        """
        Function working with Tensors instead of instances for proper
        computing in the Tensorflow graph.
        """
        raise NotImplemented()

    def _anchor_target_layer_np(self, pretrained_shape, gt_boxes, im_size, all_anchors):

        if self._debug:
            np.random.seed(0)

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
            (all_anchors[:, 2] < im_size[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_size[0] + self._allowed_border)    # height
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
        num_fg = int(self._foreground_fraction * self._minibatch_size)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self._minibatch_size - np.sum(labels == 1)
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
        """Compute bounding-box regression targets for an image.

        Regression targets are the needed adjustments to transform the bounding
        boxes of the anchors to each respective ground truth box.

        For details on how this adjustment is implemented look a the
        `bbox_transform` module.

        Args:
            boxes: Numpy array with the anchors bounding boxes.
                Its shape should be (total_bboxes, 4) where total_bboxes
                is the number of anchors available in the batch.
            groundtruth_boxes: Numpy array with the groundtruth_boxes.
                Its shape should be (total_bboxes, 4) or (total_bboxes, 5)
                depending if the label is included or not. Either way, we don't
                need it.
        """
        return encode(
            boxes, groundtruth_boxes[:, :4]
        ).astype(np.float32, copy=False)
