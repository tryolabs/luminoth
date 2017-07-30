import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_transform_tf import bbox_decode


class RCNNProposal(snt.AbstractModule):
    """
    Outputs final object detection proposals by
        - choosing the highest scored class for each proposal
        - applying class Non-max suppression
        - Top N filtering (TODO)

    """
    def __init__(self, num_classes, config, name='rcnn_proposal'):
        super(RCNNProposal, self).__init__(name=name)
        self._num_classes = num_classes

        self._class_max_detections = config.class_max_detections
        self._class_nms_threshold = config.class_nms_threshold
        # TODO: Total max detections implementation
        self._total_max_detections = config.total_max_detections

    def _build(self, proposals, bbox_pred, cls_prob, im_shape):
        """
        Args:
            proposals:
                Shape (num_proposals, 5).
                Where num_proposals es <= POST_NMS_TOP_N (We don't know beforehand)
            bbox_pred:
                Shape (num_proposals, 4 * num_classes)
                A bounding box delta prediction for each proposal for each class.
            cls_prob:
                Shape (num_proposals, num_classes + 1)
                A softmax probability for each proposal where the idx = 0 is the
                background class (which we should ignore).

        Returns:
            objects:
                Shape (final_num_proposals, 4)
                Where final_num_proposals is unknown before-hand (it depends on NMS)
                And the 4-length Tensor for each corresponds to:
                (x_min, y_min, x_max, y_max).
            objects_label:
                Shape (final_num_proposals, 1)
            objects_label_prob:
                Shape (final_num_proposals, 1)

        """

        # remove batch_id from proposals (TODO: Multibatch support?, batch_id should be called image_idx?)
        with tf.control_dependencies([tf.equal(tf.shape(proposals)[-1], 5)]):
            proposals = tf.slice(proposals, [0, 1], [-1, -1])

        # First we want get the most probable label for each proposal
        # We still have the background on idx 0 so we substract 1 to the idxs.
        proposal_label = tf.argmax(cls_prob, axis=1) - 1
        # Get the probability for the selected label for each proposal.
        proposal_label_prob = tf.reduce_max(cls_prob, axis=1)

        # We are going to use only the non-background proposals.
        proposal_filter = tf.greater_equal(proposal_label, 0)

        with tf.control_dependencies([tf.assert_equal(tf.shape(proposals)[0], tf.shape(bbox_pred)[0])]):
            # Filter all tensors for getting all non-background proposals.
            proposals = tf.boolean_mask(
                proposals, proposal_filter)
            proposal_label = tf.boolean_mask(
                proposal_label, proposal_filter)
            proposal_label_prob = tf.boolean_mask(
                proposal_label_prob, proposal_filter)
            bbox_pred = tf.boolean_mask(
                bbox_pred, proposal_filter)

        # Create one hot with labels for using it to filter bbox_predictions.
        label_one_hot = tf.one_hot(proposal_label, depth=self._num_classes)
        # Flatten label_one_hot to get (num_non_background_proposals * num_classes, 1) for filtering.
        label_one_hot_flatten = tf.cast(tf.reshape(label_one_hot, [-1]), tf.bool)
        # Flatten bbox_predictions getting (num_non_background_proposals * num_classes, 4).
        bbox_pred_flatten = tf.reshape(bbox_pred, [-1, 4])

        with tf.control_dependencies([
                tf.assert_equal(
                    tf.shape(bbox_pred_flatten)[0],
                    tf.shape(label_one_hot_flatten)[0])]):
            # Control same number of dimensions between bbox and mask.
            bbox_pred = tf.boolean_mask(
                bbox_pred_flatten, label_one_hot_flatten)

        objects_unfiltered = bbox_decode(proposals, bbox_pred)

        x_min, y_min, x_max, y_max = tf.unstack(objects_unfiltered, axis=1)

        # Clip boxes
        image_shape = tf.cast(im_shape, tf.float32)
        x_min = tf.maximum(tf.minimum(x_min, image_shape[1] - 1), 0.)
        y_min = tf.maximum(tf.minimum(y_min, image_shape[0] - 1), 0.)
        x_max = tf.maximum(tf.minimum(x_max, image_shape[1] - 1), 0.)
        y_max = tf.maximum(tf.minimum(y_max, image_shape[0] - 1), 0.)

        objects_tf = tf.stack([y_min, x_min, y_max, x_max], axis=1)

        selected_boxes = []
        selected_probs = []
        selected_labels = []
        for class_id in range(self._num_classes):
            class_filter = tf.equal(proposal_label, class_id)
            class_objects_tf = tf.boolean_mask(objects_tf, class_filter)
            class_prob = tf.boolean_mask(proposal_label_prob, class_filter)

            class_selected_idx = tf.image.non_max_suppression(
                class_objects_tf, class_prob, self._class_max_detections,
                iou_threshold=self._class_nms_threshold
            )

            class_objects_tf = tf.gather(class_objects_tf, class_selected_idx)
            class_prob = tf.gather(class_prob, class_selected_idx)

            selected_boxes.append(class_objects_tf)
            selected_probs.append(class_prob)
            selected_labels.append(
                tf.tile([class_id], [tf.shape(class_selected_idx)[0]])
            )

        objects_tf = tf.concat(selected_boxes, axis=0)
        y_min, x_min, y_max, x_max = tf.unstack(objects_tf, axis=1)
        objects = tf.stack([x_min, y_min, x_max, y_max], axis=1)
        proposal_label = tf.concat(selected_labels, axis=0)
        proposal_label_prob = tf.concat(selected_probs, axis=0)

        return {
            'objects_unfiltered': objects_unfiltered,
            'objects': objects,
            'proposal_label': proposal_label,
            'proposal_label_prob': proposal_label_prob,
            'selected_boxes': selected_boxes,
            'selected_probs': selected_probs,
            'selected_labels': selected_labels,
        }
