import tensorflow as tf
import sonnet as snt

from luminoth.utils.bbox_overlap import bbox_overlap_tf
from luminoth.utils.bbox_transform_tf import encode


class RetinaTarget(snt.AbstractModule):
    """Compute targets.
    """

    def __init__(self, config, num_classes, seed=None, name='retina_target'):
        super(RetinaTarget, self).__init__(name=name)

        # IoU lower threshold with a ground truth box to be considered that
        # specific class.
        self._foreground_threshold = config.foreground_threshold
        # High and low treshold to be considered background.
        self._background_threshold_high = config.background_threshold_high
        self._background_threshold_low = config.background_threshold_low

        self._num_classes = num_classes
        self._config = config
        self._seed = seed

    def _build(self, anchors, gt_boxes):
        """
        Args:
            anchors: (num_anchors, 4)
            gt_boxes: (num_gt_boxes, 4)

        Returns:
            Tuple (anchors_label, bbox_target)
                anchors_label: (num_anchors,)
                bbox_target: (num_anchors, 4)
        """
        # TODO: either delete this comment or properly handle batch_ids.
        # Remove batch id from anchors.
        anchors = tf.to_float(anchors)

        overlaps = bbox_overlap_tf(anchors, gt_boxes[:, :4])
        # overlaps now contains (num_anchors, num_gt_boxes) with the IoU of
        # anchor P and ground truth box G in overlaps[P, G]

        # We are going to label each anchor based on the IoU with
        # `gt_boxes`. Start by filling the labels with -1, marking them as
        # ignored.
        anchors_label_shape = tf.gather(tf.shape(anchors), [0])
        anchors_label = tf.fill(
            dims=anchors_label_shape,
            value=-1.
        )
        # For each overlap there is three possible outcomes for labelling:
        #  if max(iou) < config.background_threshold_low then we ignore.
        #  elif max(iou) <= config.background_threshold_high then we label
        #      background.
        #  elif max(iou) > config.foreground_threshold then we label with
        #      the highest IoU in overlap.
        #
        # max_overlaps gets, for each anchor, the index in which we can
        # find the gt_box with which it has the highest overlap.
        max_overlaps = tf.reduce_max(overlaps, axis=1)

        iou_is_high_enough_for_bg = tf.greater_equal(
            max_overlaps, self._background_threshold_low
        )
        iou_is_not_too_high_for_bg = tf.less(
            max_overlaps, self._background_threshold_high
        )
        bg_condition = tf.logical_and(
            iou_is_high_enough_for_bg, iou_is_not_too_high_for_bg
        )
        anchors_label = tf.where(
            condition=bg_condition,
            x=tf.zeros_like(anchors_label, dtype=tf.float32),
            y=anchors_label
        )

        # Get the index of the best gt_box for each anchor.
        overlaps_best_gt_idxs = tf.argmax(overlaps, axis=1)
        # Having the index of the gt bbox with the best label we need to get
        # the label for each gt box and sum it one because 0 is used for
        # background.
        best_fg_labels_for_anchors = tf.add(
            tf.gather(gt_boxes[:, 4], overlaps_best_gt_idxs),
            1.
        )
        iou_is_fg = tf.greater_equal(
            max_overlaps, self._foreground_threshold
        )
        best_anchors_idxs = tf.argmax(overlaps, axis=0)

        # Set the indices in best_anchors_idxs to True, and the rest to
        # false.
        # tf.sparse_to_dense is used because we know the set of indices which
        # we want to set to True, and we know the rest of the indices
        # should be set to False. That's exactly the use case of
        # tf.sparse_to_dense.
        is_best_box = tf.sparse_to_dense(
            sparse_indices=tf.reshape(best_anchors_idxs, [-1]),
            sparse_values=True, default_value=False,
            output_shape=tf.cast(anchors_label_shape, tf.int64),
            validate_indices=False
        )
        # We update anchors_label with the value in
        # best_fg_labels_for_anchors only when the box is foreground.
        anchors_label = tf.where(
            condition=iou_is_fg,
            x=best_fg_labels_for_anchors,
            y=anchors_label
        )
        # Now we need to find the anchors that are the best for each of the
        # gt_boxes. We overwrite the previous anchors_label with this
        # because setting the best anchor for each gt_box has priority.
        best_anchors_gt_labels = tf.sparse_to_dense(
            sparse_indices=tf.reshape(best_anchors_idxs, [-1]),
            sparse_values=gt_boxes[:, 4] + 1,
            default_value=0.,
            output_shape=tf.cast(anchors_label_shape, tf.int64),
            validate_indices=False,
            name="get_right_labels_for_bestboxes"
        )
        anchors_label = tf.where(
            condition=is_best_box,
            x=best_anchors_gt_labels,
            y=anchors_label,
            name="update_labels_for_bestbox_anchors"
        )

        ######################
        #    Bbox targets    #
        ######################

        # Get the ids of the anchors that matter for bbox_target comparison.
        is_anchor_with_target = tf.greater(
            anchors_label, 0
        )
        anchors_with_target_idx = tf.where(
            condition=is_anchor_with_target
        )
        # Get the corresponding ground truth box only for the anchors with
        # target.
        gt_boxes_idxs = tf.gather(
            overlaps_best_gt_idxs,
            anchors_with_target_idx
        )
        # Get the values of the ground truth boxes.
        anchors_gt_boxes = tf.gather_nd(
            gt_boxes[:, :4], gt_boxes_idxs
        )
        # We create the same array but with the anchors
        anchors_with_target = tf.gather_nd(
            anchors,
            anchors_with_target_idx
        )
        # We create our targets with bbox_transform
        bbox_targets_nonzero = encode(
            anchors_with_target,
            anchors_gt_boxes,
        )
        # TODO: We should normalize it in order for bbox_targets to have zero
        # mean and unit variance according to the paper.

        # We unmap these values, filling with zeroes to get the shape of
        # anchors
        bbox_targets = tf.scatter_nd(
            indices=anchors_with_target_idx,
            updates=bbox_targets_nonzero,
            shape=tf.cast(tf.shape(anchors), tf.int64)
        )

        return anchors_label, bbox_targets
