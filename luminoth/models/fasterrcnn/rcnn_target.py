import tensorflow as tf
import sonnet as snt
import numpy as np

from luminoth.utils.bbox_transform import encode, unmap
from luminoth.utils.bbox_overlap import bbox_overlap


class RCNNTarget(snt.AbstractModule):
    """Generate RCNN target tensors for both probabilities and bounding boxes.

    Targets for RCNN are based upon the results of the RPN, this can get tricky
    in the sense that RPN results might not be the best and it might not be
    possible to have the ideal amount of targets for all the available ground
    truth boxes.

    There are two types of targets, class targets and bounding box targets.

    Class targets are used both for background and foreground, while bounding
    box targets are only used for foreground (since it's not possible to create
    a bounding box of "background objects").

    A minibatch size determines how many targets are going to be generated and
    how many are going to be ignored. RCNNTarget is responsable for choosing
    which proposals and corresponding targets are included in the minibatch and
    which ones are completly ignored.

    TODO: Estimate performance degradation when running py_func.
    """
    def __init__(self, num_classes, config, debug=False, name='rcnn_proposal'):
        """
        Args:
            num_classes: Number of possible classes.
            config: Configuration object for RCNNTarget.
        """
        super(RCNNTarget, self).__init__(name=name)
        self._num_classes = num_classes
        # Ratio of foreground vs background for the minibatch.
        self._foreground_fraction = config.foreground_fraction
        self._minibatch_size = config.minibatch_size
        # IoU lower threshold with a ground truth box to be considered that
        # specific class.
        self._foreground_threshold = config.foreground_threshold
        # High and low treshold to be considered background.
        self._background_threshold_high = config.background_threshold_high
        self._background_threshold_low = config.background_threshold_low
        self._debug = debug

        if self._debug:
            tf.logging.warning(
                'Using RCNN Target in debug mode makes random seed '
                'to be fixed at 0.')

    def _build(self, proposals, gt_boxes):
        """
        RCNNTarget is implemented in numpy and attached to the computation
        graph using the `tf.py_func` operation.

        Ideally we should implement it as pure TensorFlow operation to avoid
        having to execute it in the CPU.

        Args:
            proposals: A Tensor with the RPN bounding boxes proposals.
                The shape of the Tensor is (num_proposals, 5), where the first
                of the 5 values for each proposal is the batch number.
            gt_boxes: A Tensor with the ground truth boxes for the image.
                The shape of the Tensor is (num_gt, 5), having the truth label
                as the last value for each box.
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
        (proposals_label, bbox_targets) = tf.py_func(
            self.proposal_target_layer,
            [proposals, gt_boxes],
            [tf.float32, tf.float32],
            stateful=False,
            name='proposal_target_layer'
         )

        return proposals_label, bbox_targets

    def proposal_target_layer(self, proposals, gt_boxes):
        """
        Numpy implementation for _build.
        """
        if self._debug:
            np.random.seed(0)  # TODO: For reproducibility.

        # Remove batch id from proposals
        proposals = proposals[:, 1:]

        overlaps = bbox_overlap(
            # We need to use float and ascontiguousarray because of Cython
            # implementation of bbox_overlap
            np.ascontiguousarray(proposals, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float)
        )

        # overlaps returns (num_proposals, num_gt_boxes) with the IoU of
        # proposal P and ground truth box G in overlaps[P, G]

        # We are going to label each proposal based on the IoU with `gt_boxes`.
        # Start by filling the labels with -1, marking them as ignored.
        proposals_label = np.empty((proposals.shape[0], ), dtype=np.float32)
        proposals_label.fill(-1)

        # For each overlap there is three possible outcomes for labelling:
        #  if max(iou) < 0.1 then we ignore
        #  elif max(iou) <= 0.5 then we label background
        #  elif max(iou) > 0.5 then we label with the highest IoU in overlap
        max_overlaps = overlaps.max(axis=1)

        # Label background
        proposals_label[
            (max_overlaps > self._background_threshold_low) &
            (max_overlaps < self._background_threshold_high)
        ] = 0

        # Filter proposal that have labels
        overlaps_with_label = max_overlaps >= self._foreground_threshold
        # Get label for proposal with labels
        overlaps_best_label = overlaps.argmax(axis=1)
        # Having the index of the gt bbox with the best label we need to get
        # the label for each gt box and sum it one because 0 is used for
        # background.
        # we only assign to proposals with `overlaps_with_label`.
        proposals_label[overlaps_with_label] = (
            gt_boxes[:, 4][overlaps_best_label] + 1
        )[overlaps_with_label]

        # Finally we get the closest proposal for each ground truth box and
        # mark it as positive.
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        proposals_label[gt_argmax_overlaps] = gt_boxes[:, 4] + 1

        # proposals_label now has [0, num_classes + 1] for proposals we are
        # going to use and -1 for the ones we should ignore.

        # Now we subsample labels and mark them as -1 in order to ignore them.
        # Our batch of N should be: F% foreground (label > 0)
        num_fg = int(self._foreground_fraction * self._minibatch_size)
        fg_inds = np.where(proposals_label >= 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False
            )
            # We disable them with their negatives just to be able to debug
            # down the road.
            proposals_label[disable_inds] = - proposals_label[disable_inds]

        if len(fg_inds) < num_fg:
            # When our foreground samples are not as much as we would like them
            # to be, we log a warning message.
            tf.logging.warning(
                'We\'ve got only {} foreground samples instead of {}.'.format(
                    len(fg_inds), num_fg
                )
            )

        # subsample negative labels
        num_bg = self._minibatch_size - np.sum(proposals_label >= 1)
        bg_inds = np.where(proposals_label == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False
            )
            proposals_label[disable_inds] = -1

        """
        Next step is to calculate the proper targets for the proposals labeled
        based on the values of the ground-truth boxes.
        We have to use only the proposals labeled >= 1, each matching with
        the proper gt_boxes
        """

        # Get the ids of the proposals that matter for bbox_target comparisson.
        proposal_with_target_idx = np.nonzero(proposals_label > 0)[0]

        # Get top gt_box for every proposal, top_gt_idx shape (1000,) with
        # values < gt_boxes.shape[0]
        top_gt_idx = overlaps.argmax(axis=1)

        # Get the corresponding ground truth box only for the proposals with
        # target.
        gt_boxes_ids = top_gt_idx[proposal_with_target_idx]

        # Get the values of the ground truth boxes. This is shaped
        # (num_proposals, 5) because we also have the label.
        proposals_gt_boxes = gt_boxes[gt_boxes_ids]

        # We create the same array but with the proposals
        proposals_with_target = proposals[proposal_with_target_idx]

        # We create our targets with bbox_transform
        bbox_targets = encode(
            proposals_with_target, proposals_gt_boxes
        )
        # TODO: We should normalize it in order for bbox_targets to have zero
        # mean and unit variance according to the paper.

        # We unmap targets to proposal_labels (containing the length of
        # proposals)
        bbox_targets = unmap(
            bbox_targets, proposals_label.shape[0], proposal_with_target_idx,
            fill=0
        )

        return proposals_label, bbox_targets
