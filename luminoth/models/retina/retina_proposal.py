import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_transform_tf import change_order


class RetinaProposal(snt.AbstractModule):
    """Postprocessing of the output to get the proposals.

    We apply class-based NMS and filter undesired proposals
    (low probability, etc.)
    """
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

        self._min_prob_threshold = config.min_prob_threshold

    def _build(self, cls_score, proposals, all_anchors):
        """
        Args:
            cls_score: (num_proposals,)
            proposals: (num_proposals, 4)
            all_anchors: (num_proposals, 4)

        Returns:
            objects: (num_objects, 4)
            proposal_label: (num_objects,)
            proposal_label_prob: (num_objects,)
            proposal_label_score: (num_objects,)
        """
        # First we want get the most probable label for each proposal
        # We still have the background on idx 0 so we subtract 1 to the idxs.
        proposal_label = tf.argmax(cls_score, axis=1, name='label_argmax') - 1

        # We are going to use only the non-background proposals.
        non_background_filter = tf.greater_equal(proposal_label, 0)
        # TODO: optimize this. We're doing softmax on the same scores several
        # times per step.
        cls_prob = tf.nn.softmax(cls_score)
        prob_filter = tf.greater(
            tf.reduce_max(cls_prob, axis=1, name='reduce_probs'),
            self._min_prob_threshold,
            name='get_prob_mask'
        )

        proposal_filter = tf.logical_and(
            non_background_filter, prob_filter, name='combined_filters'
        )

        total_proposals = tf.shape(proposals)[0]

        proposals = tf.boolean_mask(
            proposals, proposal_filter, name='mask_proposals'
        )
        proposal_label = tf.boolean_mask(
            proposal_label, proposal_filter, name='mask_labels'
        )
        cls_score = tf.boolean_mask(
            cls_score, proposal_filter, name='mask_scores'
        )

        filtered_proposals = tf.shape(proposals)[0]

        tf.summary.scalar(
            'background_proposals',
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
            this_class_score = tf.boolean_mask(cls_score, class_filter)
            this_class_score = this_class_score[:, class_id + 1]
            this_class_score = tf.reshape(this_class_score, [-1])
            this_class_score = tf.Print(
                this_class_score,
                [
                    tf.shape(class_objects_tf), tf.shape(this_class_score)
                ],
                message='SHP_OBJS, SHP_SCORE: '
            )

            # Apply class NMS.
            class_selected_idx = tf.image.non_max_suppression(
                class_objects_tf, this_class_score,
                self._class_max_detections,
                iou_threshold=self._class_nms_threshold
            )

            # Using NMS resulting indices, gather values from Tensors.
            class_objects_tf = tf.gather(class_objects_tf, class_selected_idx)
            this_class_score = tf.gather(
                this_class_score, class_selected_idx)

            # We append values to a regular list which will later be transform
            # to a proper Tensor.
            selected_boxes.append(class_objects_tf)
            selected_probs.append(tf.nn.softmax(this_class_score))
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
