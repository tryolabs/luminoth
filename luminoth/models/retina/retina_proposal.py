import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_transform_tf import change_order


class RetinaProposal(snt.AbstractModule):

    def __init__(self, config, num_classes, debug=False,
                 name='retina_proposal'):
        super(RetinaProposal, self).__init__(name=name)

        self._num_classes = num_classes

        # Max number of object detections per class.
        self._class_max_detections = config.class_max_detections
        # NMS intersection over union threshold to be used for classes.
        self._class_nms_threshold = config.class_nms_threshold
        # Maximum number of detections to return.
        self._total_max_detections = config.total_max_detections
        # Threshold probability
        self._min_prob_threshold = config.min_prob_threshold or 0.0

    def _build(self, cls_prob, cls_score, proposals, all_anchors, im_shape):
        # First we want get the most probable label for each proposal
        # We still have the background on idx 0 so we subtract 1 to the idxs.
        proposal_label = tf.argmax(cls_prob, axis=1, name='label_argmax') - 1
        # Get the probability for the selected label for each proposal.
        proposal_label_prob = tf.reduce_max(cls_prob, axis=1, name='max_label')

        # We are going to use only the non-background proposals.
        non_background_filter = tf.greater_equal(proposal_label, 0)
        # Filter proposals with less than threshold probability.
        min_prob_filter = tf.greater_equal(
            proposal_label_prob, self._min_prob_threshold
        )
        proposal_filter = tf.logical_and(
            non_background_filter, min_prob_filter
        )

        total_proposals = tf.shape(proposals)[0]

        proposals = tf.boolean_mask(
            proposals, proposal_filter, name='mask_proposals')
        proposal_label = tf.boolean_mask(
            proposal_label, proposal_filter, name='mask_labels')
        proposal_label_prob = tf.boolean_mask(
            proposal_label_prob, proposal_filter, name='mask_probs')

        filtered_proposals = tf.shape(proposals)[0]

        tf.summary.scalar(
            'background_or_low_prob_proposals',
            total_proposals - filtered_proposals,
            ['retina']
        )

        # We have to use the TensorFlow's bounding box convention to use the
        # included function for NMS.
        # After gathering results we should normalize it back.
        objects_tf = change_order(proposals)

        selected_boxes = []
        selected_probs = []
        selected_labels = []
        # For each class we want to filter those objects and apply NMS to them.
        for class_id in range(self._num_classes):
            # Filter objects Tensors with class.
            class_filter = tf.equal(proposal_label, class_id)
            class_objects_tf = tf.boolean_mask(objects_tf, class_filter)
            class_prob = tf.boolean_mask(proposal_label_prob, class_filter)

            # Apply class NMS.
            class_selected_idx = tf.image.non_max_suppression(
                class_objects_tf, class_prob, self._class_max_detections,
                iou_threshold=self._class_nms_threshold
            )

            # Using NMS resulting indices, gather values from Tensors.
            class_objects_tf = tf.gather(class_objects_tf, class_selected_idx)
            class_prob = tf.gather(class_prob, class_selected_idx)

            # We append values to a regular list which will later be transform
            # to a proper Tensor.
            selected_boxes.append(class_objects_tf)
            selected_probs.append(class_prob)
            # In the case of the class_id, since it is a loop on classes, we
            # already have a fixed class_id. We use `tf.tile` to create that
            # Tensor with the total number of indices returned by the NMS.
            selected_labels.append(
                tf.tile([class_id], [tf.shape(class_selected_idx)[0]])
            )

        # We use concat (axis=0) to generate a Tensor where the rows are
        # stacked on top of each other
        objects_tf = tf.concat(selected_boxes, axis=0)
        # Return to the original convention.
        objects = change_order(objects_tf)
        proposal_label = tf.concat(selected_labels, axis=0)
        proposal_label_prob = tf.concat(selected_probs, axis=0)

        # Get topK detections of all classes.
        k = tf.minimum(
            self._total_max_detections,
            tf.shape(proposal_label_prob)[0]
        )
        top_k = tf.nn.top_k(proposal_label_prob, k=k)
        top_k_proposal_label_prob = top_k.values
        top_k_objects = tf.gather(objects, top_k.indices)
        top_k_proposal_label = tf.gather(proposal_label, top_k.indices)
        top_k_score = tf.gather(cls_score, top_k.indices)

        return {
            'objects': top_k_objects,
            'proposal_label': top_k_proposal_label,
            'proposal_label_prob': top_k_proposal_label_prob,
            'proposal_label_score': top_k_score,
            'selected_boxes': selected_boxes,
            'selected_probs': selected_probs,
            'selected_labels': selected_labels,
        }
