import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_transform_tf import decode, clip_boxes, change_order


class SSDProposal(snt.AbstractModule):
    """Transforms anchors and SSD predictions into object proposals.

    Using the fixed anchors and the SSD predictions for both classification
    and regression (adjusting the bounding box), we return a list of proposals
    with assigned class.

    In the process it tries to remove duplicated suggestions by applying non
    maximum suppresion (NMS).

    We apply NMS because the way object detectors are usually scored is by
    treating duplicated detections (multiple detections that overlap the same
    ground truth value) as false positive. It is resonable to assume that there
    may exist such case that applying NMS is completly unnecesary.

    Besides applying NMS it also filters the top N results, both for classes
    and in general. These values are easily modifiable in the configuration
    files.
    """
    def __init__(self, num_anchors, num_classes, config, debug=False,
                 name='proposal_layer'):
        super(SSDProposal, self).__init__(name=name)
        self._num_anchors = num_anchors
        self._num_classes = num_classes

        # Threshold to use for NMS.
        self._class_nms_threshold = config.class_nms_threshold
        # Max number of proposals detections per class.
        self._class_max_detections = config.class_max_detections
        # Maximum number of detections to return.
        self._total_max_detections = config.total_max_detections
        self._min_prob_threshold = config.min_prob_threshold or 0.0

        self._filter_outside_anchors = config.filter_outside_anchors
        self._debug = debug

    def _build(self, cls_prob, loc_pred, all_anchors, im_shape):
        """
        Args:
            cls_prob: A softmax probability for each anchor where the idx = 0
                is the background class (which we should ignore).
                Shape (total_anchors, num_classes + 1)
            loc_pred: A Tensor with the regression output for each anchor.
                Its shape should be (total_anchors, 4).
            all_anchors: A Tensor with the anchors bounding boxes of shape
                (total_anchors, 4), having (x_min, y_min, x_max, y_max) for
                each anchor.
            im_shape: A Tensor with the image shape in format (height, width).
        Returns:
            prediction_dict with the following keys:
                raw_proposals: The raw proposals i.e. the anchors adjusted
                    using loc_pred.
                proposals: The proposals of the network after appling some
                    filters like negative area; and NMS. It's shape is
                    (final_num_proposals, 4), where final_num_proposals is
                    unknown before-hand (it depends on NMS).
                    The 4-length Tensor for each corresponds to:
                    (x_min, y_min, x_max, y_max).
                proposal_label: It's shape is (final_num_proposals,)
                proposal_label_prob: It's shape is (final_num_proposals,)
        """
        # First we want get the most probable label for each anchor
        # We still have the background on idx 0 so we substract 1 to the idxs,
        # so the background label now is -1.
        proposal_label = tf.argmax(cls_prob, axis=1) - 1
        # Get the probability for the selected label for each proposal.
        proposal_label_prob = tf.reduce_max(cls_prob, axis=1)

        # We are going to use only the non-background anchors.
        non_background_filter = tf.greater_equal(proposal_label, 0)
        # Filter anchors with less than threshold probability.
        min_prob_filter = tf.greater_equal(
            proposal_label_prob, self._min_prob_threshold
        )
        proposal_filter = tf.logical_and(
            non_background_filter, min_prob_filter
        )

        total_anchors = tf.shape(all_anchors)[0]

        equal_shapes = tf.assert_equal(
            tf.shape(all_anchors)[0], tf.shape(loc_pred)[0]
        )
        with tf.control_dependencies([equal_shapes]):
            # Filter all tensors for getting all non-background anchors.
            all_anchors = tf.boolean_mask(
                all_anchors, proposal_filter)
            proposal_label = tf.boolean_mask(
                proposal_label, proposal_filter)
            proposal_label_prob = tf.boolean_mask(
                proposal_label_prob, proposal_filter)
            loc_pred = tf.boolean_mask(
                loc_pred, proposal_filter)

        filtered_anchors = tf.shape(all_anchors)[0]
        tf.summary.scalar(
             'background_or_low_prob_anchors',
             total_anchors - filtered_anchors, ['ssd_proposal']
         )

        # Using the loc_pred and the anchors we generate the proposals.
        raw_proposals = decode(all_anchors, loc_pred)
        # Clip boxes to image.
        clipped_proposals = clip_boxes(raw_proposals, im_shape)

        # Filter proposals that have an non-valid area.
        (x_min, y_min, x_max, y_max) = tf.unstack(clipped_proposals, axis=1)
        proposal_filter = tf.greater(
            tf.maximum(x_max - x_min, 0.0) * tf.maximum(y_max - y_min, 0.0),
            0.0
        )

        total_raw_proposals = tf.shape(raw_proposals)[0]

        proposals = tf.boolean_mask(
            clipped_proposals, proposal_filter)
        proposal_label = tf.boolean_mask(
            proposal_label, proposal_filter)
        proposal_label_prob = tf.boolean_mask(
            proposal_label_prob, proposal_filter)

        total_proposals = tf.shape(proposals)[0]

        tf.summary.scalar(
            'invalid_proposals',
            total_proposals - total_raw_proposals, ['ssd']
        )

        tf.summary.scalar(
            'valid_proposals_ratio',
            tf.cast(total_anchors, tf.float32) /
            tf.cast(total_proposals, tf.float32), ['ssd']
        )

        # We have to use the TensorFlow's bounding box convention to use the
        # included function for NMS.
        # After gathering results we should normalize it back.
        proposals_tf = change_order(proposals)

        selected_boxes = []
        selected_probs = []
        selected_labels = []
        # For each class we want to filter those proposals and apply NMS.
        for class_id in range(self._num_classes):
            # Filter proposals Tensors with class.
            class_filter = tf.equal(proposal_label, class_id)
            class_proposal_tf = tf.boolean_mask(proposals_tf, class_filter)
            class_prob = tf.boolean_mask(proposal_label_prob, class_filter)

            # Apply class NMS.
            class_selected_idx = tf.image.non_max_suppression(
                class_proposal_tf, class_prob, self._class_max_detections,
                iou_threshold=self._class_nms_threshold
            )

            # Using NMS resulting indices, gather values from Tensors.
            class_proposal_tf = tf.gather(
                class_proposal_tf, class_selected_idx)
            class_prob = tf.gather(class_prob, class_selected_idx)

            # We append values to a regular list which will later be transform
            # to a proper Tensor.
            selected_boxes.append(class_proposal_tf)
            selected_probs.append(class_prob)
            # In the case of the class_id, since it is a loop on classes, we
            # already have a fixed class_id. We use `tf.tile` to create that
            # Tensor with the total number of indices returned by the NMS.
            selected_labels.append(
                tf.tile([class_id], [tf.shape(class_selected_idx)[0]])
            )

        # We use concat (axis=0) to generate a Tensor where the rows are
        # stacked on top of each other
        proposals_tf = tf.concat(selected_boxes, axis=0)
        # Return to the original convention.
        proposals = change_order(proposals_tf)
        proposal_label = tf.concat(selected_labels, axis=0)
        proposal_label_prob = tf.concat(selected_probs, axis=0)

        # Get topK detections of all classes.
        k = tf.minimum(
            self._total_max_detections,
            tf.shape(proposal_label_prob)[0]
        )
        top_k = tf.nn.top_k(proposal_label_prob, k=k)
        top_k_proposal_label_prob = top_k.values
        top_k_proposals = tf.gather(proposals, top_k.indices)
        top_k_proposal_label = tf.gather(proposal_label, top_k.indices)

        return {
            'objects': top_k_proposals,
            'labels': top_k_proposal_label,
            'probs': top_k_proposal_label_prob
        }
