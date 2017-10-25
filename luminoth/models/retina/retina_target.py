import tensorflow as tf
import sonnet as snt

from luminoth.utils.bbox_overlap import bbox_overlap_tf
from luminoth.utils.bbox_transform_tf import encode


class RetinaTarget(snt.AbstractModule):

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

    def _build(self, proposals, gt_boxes):
        # TODO: either delete this comment or properly handle batch_ids.
        # Remove batch id from proposals.
        # proposals = proposals[:, 1:]

        overlaps = bbox_overlap_tf(proposals, gt_boxes[:, :4])
        # overlaps now contains (num_proposals, num_gt_boxes) with the IoU of
        # proposal P and ground truth box G in overlaps[P, G]

        # We are going to label each proposal based on the IoU with
        # `gt_boxes`. Start by filling the labels with -1, marking them as
        # ignored.
        proposals_label_shape = tf.gather(tf.shape(proposals), [0])
        proposals_label = tf.fill(
            dims=proposals_label_shape,
            value=-1.
        )
        # For each overlap there is three possible outcomes for labelling:
        #  if max(iou) < config.background_threshold_low then we ignore.
        #  elif max(iou) <= config.background_threshold_high then we label
        #      background.
        #  elif max(iou) > config.foreground_threshold then we label with
        #      the highest IoU in overlap.
        #
        # max_overlaps gets, for each proposal, the index in which we can
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
        proposals_label = tf.where(
            condition=bg_condition,
            x=tf.zeros_like(proposals_label, dtype=tf.float32),
            y=proposals_label
        )

        # Get the index of the best gt_box for each proposal.
        overlaps_best_gt_idxs = tf.argmax(overlaps, axis=1)
        # Having the index of the gt bbox with the best label we need to get
        # the label for each gt box and sum it one because 0 is used for
        # background.
        best_fg_labels_for_proposals = tf.add(
            tf.gather(gt_boxes[:, 4], overlaps_best_gt_idxs),
            1.
        )
        iou_is_fg = tf.greater_equal(
            max_overlaps, self._foreground_threshold
        )
        best_proposals_idxs = tf.argmax(overlaps, axis=0)

        # Set the indices in best_proposals_idxs to True, and the rest to
        # false.
        # tf.sparse_to_dense is used because we know the set of indices which
        # we want to set to True, and we know the rest of the indices
        # should be set to False. That's exactly the use case of
        # tf.sparse_to_dense.
        is_best_box = tf.sparse_to_dense(
            sparse_indices=tf.reshape(best_proposals_idxs, [-1]),
            sparse_values=True, default_value=False,
            output_shape=tf.cast(proposals_label_shape, tf.int64),
            validate_indices=False
        )
        # We update proposals_label with the value in
        # best_fg_labels_for_proposals only when the box is foreground.
        proposals_label = tf.where(
            condition=iou_is_fg,
            x=best_fg_labels_for_proposals,
            y=proposals_label
        )
        # Now we need to find the proposals that are the best for each of the
        # gt_boxes. We overwrite the previous proposals_label with this
        # because setting the best proposal for each gt_box has priority.
        best_proposals_gt_labels = tf.sparse_to_dense(
            sparse_indices=tf.reshape(best_proposals_idxs, [-1]),
            sparse_values=gt_boxes[:, 4] + 1,
            default_value=0.,
            output_shape=tf.cast(proposals_label_shape, tf.int64),
            validate_indices=False,
            name="get_right_labels_for_bestboxes"
        )
        proposals_label = tf.where(
            condition=is_best_box,
            x=best_proposals_gt_labels,
            y=proposals_label,
            name="update_labels_for_bestbox_proposals"
        )

        ######################
        #    Bbox targets    #
        ######################

        # Get the ids of the proposals that matter for bbox_target comparisson.
        is_proposal_with_target = tf.greater(
            proposals_label, 0
        )
        proposals_with_target_idx = tf.where(
            condition=is_proposal_with_target
        )
        # Get the corresponding ground truth box only for the proposals with
        # target.
        gt_boxes_idxs = tf.gather(
            overlaps_best_gt_idxs,
            proposals_with_target_idx
        )
        # Get the values of the ground truth boxes.
        proposals_gt_boxes = tf.gather_nd(
            gt_boxes[:, :4], gt_boxes_idxs
        )
        # We create the same array but with the proposals
        proposals_with_target = tf.gather_nd(
            proposals,
            proposals_with_target_idx
        )
        # We create our targets with bbox_transform
        bbox_targets_nonzero = encode(
            proposals_with_target,
            proposals_gt_boxes,
        )
        # TODO: We should normalize it in order for bbox_targets to have zero
        # mean and unit variance according to the paper.

        # We unmap targets to proposal_labels (containing the length of
        # proposals)
        bbox_targets = tf.scatter_nd(
            indices=proposals_with_target_idx,
            updates=bbox_targets_nonzero,
            shape=tf.cast(tf.shape(proposals), tf.int64)
        )

        return proposals_label, bbox_targets
