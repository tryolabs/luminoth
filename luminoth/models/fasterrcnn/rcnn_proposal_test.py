import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn.rcnn_proposal import RCNNProposal
from luminoth.utils.bbox_transform_tf import encode


class RCNNProposalTest(tf.test.TestCase):

    def setUp(self):
        super(RCNNProposalTest, self).setUp()

        self._num_classes = 3
        self._batch_number = 1

        self._image_shape = (1440, 900)
        self._config = EasyDict({
            'class_max_detections': 100,
            'class_nms_threshold': 0.6,
            'total_max_detections': 300,
        })

        self._equality_delta = 1e-03

        self._shared_model = RCNNProposal(self._num_classes, self._config)

    def _run_rcnn_proposal(self, model, proposals, bbox_pred, cls_prob,
                           image_shape=None):
        if image_shape is None:
            image_shape = self._image_shape
        rcnn_proposal_net = model(proposals, bbox_pred, cls_prob, image_shape)
        with tf.Session() as sess:
            return sess.run(rcnn_proposal_net)

    def _compute_tf_graph(self, graph):
        with tf.Session() as sess:
            return sess.run(graph)

    def _get_bbox_pred(self, proposed_boxes, gt_boxes_per_class):
        """Computes valid bbox_preds from proposals and gt_boxes for each class.

        Arguments:
            proposed_boxes: Tensor with shape (num_proposals, 5).
            gt_boxes_per_class: Tensor holding the ground truth boxes for each
                class. Has shape (num_classes, num_gt_boxes_per_class, 4).

        Returns:
            A tensor with shape (num_proposals, num_classes * 4), holding the
            correct bbox_preds.
        """
        def bbox_encode(gt_boxes):
            return encode(proposed_boxes[:, 1:], gt_boxes)
        bbox_pred_tensor = tf.map_fn(bbox_encode, gt_boxes_per_class,
                                     dtype=tf.float32)
        bbox_pred_list = tf.unstack(bbox_pred_tensor)
        return tf.concat(bbox_pred_list, 1)

    def testBackgroundFilter(self):
        """Tests that we're not returning an object when a proposal is background.

        This includes two sub-tests. One case in which there is a foreground
        proposal, and one in which all proposals are background. We use the
        same proposed_boxes and gt_boxes, but change the cls_prob.
        """
        proposed_boxes = tf.constant([
            (self._batch_number, 85, 500, 730, 590),
            (self._batch_number, 50, 500, 70, 530),
            (self._batch_number, 700, 570, 740, 598),
        ])
        gt_boxes_per_class = tf.constant([
            [(101, 101, 201, 249)],
            [(200, 502, 209, 532)],
            [(86, 571, 743, 599)],
        ])
        bbox_pred = self._get_bbox_pred(proposed_boxes, gt_boxes_per_class)
        cls_prob_one_foreground = tf.constant([
            (0., .3, .3, .4),
            (1., 0., 0., 0.),
            (.35, .3, .2, .15),
        ])
        cls_prob_all_background = tf.constant([
            (.4, 0., .3, .3),
            (.8, .1, .1, 0.),
            (.7, .05, .2, .05),
        ])

        proposal_prediction_one_foreground = self._run_rcnn_proposal(
            self._shared_model,
            proposed_boxes,
            bbox_pred,
            cls_prob_one_foreground,
        )
        proposal_prediction_all_background = self._run_rcnn_proposal(
            self._shared_model,
            proposed_boxes,
            bbox_pred,
            cls_prob_all_background,
        )
        # One foregound assertion.
        # This assertion has two purposes:
        #     1. checking that we only get one object.
        #     2. checking that that object is of class 2.
        # We take this to mean we're correctly ignoring the two proposals
        # where 'background' is the highest probability class.
        self.assertAllClose(proposal_prediction_one_foreground['objects'],
                            self._compute_tf_graph(gt_boxes_per_class)[2],
                            atol=self._equality_delta)
        # All background assertion.
        self.assertEqual(len(proposal_prediction_all_background['objects']),
                         0)

    def testNMSFilter(self):
        pass

    def testImageClipping(self):
        pass

    def testBboxPred(self):
        pass

    def testTotals(self):
        pass


if __name__ == '__main__':
    tf.test.main()
