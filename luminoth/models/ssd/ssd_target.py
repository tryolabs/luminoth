import tensorflow as tf
import sonnet as snt

from luminoth.utils.bbox_transform_tf import encode
from luminoth.utils.bbox_overlap import bbox_overlap_tf


class SSDTarget(snt.AbstractModule):
    """TODO
    """
    def __init__(self, num_classes, num_anchors,
                 config, seed=None, name='ssd_target'):
        """
        Args:
            num_classes: Number of possible classes.
            config: Configuration object for RCNNTarget.
        """
        super(SSDTarget, self).__init__(name=name)
        self._num_classes = num_classes
        # Ratio of foreground vs background for the minibatch.
        self._foreground_fraction = config.hard_negative_ratio
        self._num_anchors = tf.cast(num_anchors, tf.int32)
        # IoU lower threshold with a ground truth box to be considered that
        # specific class.
        self._foreground_threshold = config.foreground_threshold
        # High and low treshold to be considered background.
        self._background_threshold_high = config.background_threshold_high
        self._background_threshold_low = config.background_threshold_low
        self._seed = seed

    def _build(self, probs, proposals, gt_boxes, im_shape):
        """
        Args:
            proposals: A Tensor with the RPN bounding boxes proposals.
                The shape of the Tensor is (num_proposals, 4).
            gt_boxes: A Tensor with the ground truth boxes for the image.
                The shape of the Tensor is (num_gt, 5), having the truth label
                as the last value for each box.
            im_shape: (1, height, width, )
        Returns:
            proposals_label: Either a truth value of the proposals (a value
                between 0 and num_classes, with 0 being background), or -1 when
                the proposal is to be ignored in the minibatch.
                The shape of the Tensor is (num_proposals, 1).
            bbox_targets: A bounding box regression target for each of the
                proposals that have and greater than zero label. For every
                other proposal we return zeros.
                The shape of the Tensor is (num_proposals, 4).
        """

        # Proposals = all_anchors
        proposals = tf.cast(proposals, tf.float32)
        gt_boxes = tf.cast(gt_boxes, tf.float32)

        # We are going to label each proposal based on the IoU with
        # `gt_boxes`. Start by filling the labels with 0, marking them as
        # unknown (-2).
        proposals_label_shape = tf.gather(tf.shape(proposals), [0])
        proposals_label = tf.fill(
            dims=proposals_label_shape,
            value=-1.
        )

        overlaps = bbox_overlap_tf(proposals, gt_boxes[:, :4])
        # overlaps now contains (num_proposals, num_gt_boxes) with the IoU of
        # proposal P and ground truth box G in overlaps[P, G]

        # For each overlap there is two possible outcomes for labelling by now:
        #  if max(iou) > config.foreground_threshold then we label with
        #      the highest IoU in overlap.
        #  elif (config.background_threshold_low <= max(iou) <=
        #       config.background_threshold_high) we label with background (0)
        #  else we label with -1.

        # max_overlaps gets, for each proposal, the index in which we can
        # find the gt_box with which it has the highest overlap.
        max_overlaps = tf.reduce_max(overlaps, axis=1)

        # Filter proposals with negative or zero area.
        (x_min, y_min, x_max, y_max) = tf.unstack(
            proposals, axis=1
        )
        proposal_filter = tf.greater(
            tf.maximum(x_max - x_min, 0.0) * tf.maximum(y_max - y_min, 0.0),
            0.0
        )
        # We (force) reshape the filter so that we can use it as a boolean mask
        proposal_filter = tf.reshape(proposal_filter, [-1])
        # Ignore the proposals with negative area with a -1
        proposals_label = tf.where(
            condition=proposal_filter,
            x=proposals_label,
            y=tf.fill(dims=proposals_label_shape, value=-1.)
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
            default_value=-1,
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

        # Disable some backgrounds according to hard minning ratio.
        num_fg_mask = tf.greater(proposals_label, 0.0)
        num_fg = tf.cast(tf.count_nonzero(num_fg_mask), tf.float32)

        # Use the worst backgrounds (the bgs which probability of being fg is
        # the greatest).

        # Transform to one-hot vector.
        cls_target_one_hot = tf.one_hot(
            tf.cast(best_proposals_gt_labels, tf.int32), depth=self._num_classes + 1,
            name='cls_target_one_hot'
        )

        # We get cross entropy loss of each proposal.
        cross_entropy_per_proposal = (
            tf.nn.softmax_cross_entropy_with_logits(
                labels=cls_target_one_hot, logits=probs
            )
        )

        proposals_label = tf.where(
            condition=num_fg_mask,
            x=proposals_label,
            y=tf.fill(dims=proposals_label_shape, value=-1.)
        )

        # We calculate up to how many backgrounds we desire based on the
        # final number of foregrounds and the hard minning ratio.
        num_bg = tf.cast(num_fg * self._foreground_fraction, tf.int32)
        top_k_bg = tf.nn.top_k(cross_entropy_per_proposal, k=num_bg)

        set_bg = tf.sparse_to_dense(
            sparse_indices=top_k_bg.indices,
            sparse_values=True, default_value=False,
            output_shape=proposals_label_shape,
            validate_indices=False
        )

        proposals_label = tf.where(
            condition=set_bg,
            x=tf.fill(dims=proposals_label_shape, value=0.),
            y=proposals_label
        )

        """
        Next step is to calculate the proper targets for the proposals labeled
        based on the values of the ground-truth boxes.
        We have to use only the proposals labeled >= 1, each matching with
        the proper gt_boxes
        """

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

        # We unmap targets to proposal_labels (containing the length of
        # proposals)
        bbox_targets = tf.scatter_nd(
            indices=proposals_with_target_idx,
            updates=bbox_targets_nonzero,
            shape=tf.cast(tf.shape(proposals), tf.int64)
        )

        return proposals_label, bbox_targets
