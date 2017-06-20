import sonnet as snt
import tensorflow as tf
import numpy as np

from .rcnn_target import RCNNTarget

class RCNN(snt.AbstractModule):
    """RCNN """

    def __init__(self, num_classes, layer_sizes=[4096, 4096], name='rcnn'):
        super(RCNN, self).__init__(name=name)
        self._num_classes = num_classes
        self._layer_sizes = layer_sizes
        self._activation = tf.nn.relu
        self._dropout_keep_prob = 0.6
        self._instantiate_layers()

    def _instantiate_layers(self):
        with self._enter_variable_scope():

            self._layers = [
                snt.Linear(
                    layer_size,
                    name="fc_{}".format(i),
                )
                for i, layer_size in enumerate(self._layer_sizes)
            ]

            self._classifier_layer = snt.Linear(
                self._num_classes + 1, name="fc_classifier"
            )

            # TODO: Not random initializer
            self._bbox_layer = snt.Linear(
                self._num_classes * 4, name="fc_bbox"
            )

            self._rcnn_target = RCNNTarget(self._num_classes)



    def _build(self, pooled_layer, proposals, gt_boxes):
        """
        TODO: El pooled layer es el volumen con todos los ROI o es uno por cada ROI?
        TODO: Donde puedo comparar los resultados con las labels posta?
        """
        net = tf.contrib.layers.flatten(pooled_layer)
        for i, layer in enumerate(self._layers):
            net = layer(net)
            net = self._activation(net)
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        cls_score = self._classifier_layer(net)
        prob = tf.nn.softmax(cls_score)
        bbox_offsets = self._bbox_layer(net)

        proposals_targets, bbox_targets = self._rcnn_target(proposals, bbox_offsets, prob, gt_boxes)

        return {
            'cls_score': cls_score,
            'cls_prob': prob,
            'bbox_offsets': bbox_offsets,
            'cls_targets': proposals_targets,
            'bbox_offsets_targets': bbox_targets,
        }

    def loss(self, cls_prediction, cls_target, bbox_prediction, bbox_target):
        """
        Returns cost for RCNN based on:

        Args:
            cls_prediction: Class probability for each proposal.
                Shape: (num_proposals, num_classes + 1)
            cls_target: Correct class for each proposal, based in p
            bbox_prediction: Bbox class prediction adjust.
                Shape: (num_proposals, num_classes * 4)
            gt_boxes: Ground truth boxes in the dataset.
                Shape: (num_gt_boxes, 5) (4 points and 1 for true label)

        prediction_dict['refined_box_encodings'],
        prediction_dict['class_predictions_with_background'],
        prediction_dict['proposal_boxes'],
        prediction_dict['num_proposals'],
        groundtruth_boxlists,
        groundtruth_classes_with_background_list

        """
        pass