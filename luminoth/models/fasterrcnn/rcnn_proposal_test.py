import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn.rcnn_proposal import RCNNProposal
from luminoth.utils.bbox_transform_tf import encode


class RCNNProposalTest(tf.test.TestCase):

    def setUp(self):
        super(RCNNProposalTest, self).setUp()

        self._num_classes = 3
        self._image_shape = (900, 1440)
        self._config = EasyDict({
            'class_max_detections': 100,
            'class_nms_threshold': 0.6,
            'total_max_detections': 300,
            'min_prob_threshold': 0.0,
        })

        self._equality_delta = 1e-03

        self._shared_model = RCNNProposal(self._num_classes, self._config)
        tf.reset_default_graph()

    def _run_rcnn_proposal(self, model, proposals, bbox_pred, cls_prob,
                           image_shape=None):
        if image_shape is None:
            image_shape = self._image_shape
        rcnn_proposal_net = model(proposals, bbox_pred, cls_prob, image_shape)
        with self.test_session() as sess:
            return sess.run(rcnn_proposal_net)

    def _compute_tf_graph(self, graph):
        with self.test_session() as sess:
            return sess.run(graph)

    def _get_bbox_pred(self, proposed_boxes, gt_boxes_per_class):
        """Computes valid bbox_pred from proposals and gt_boxes for each class.

        Args:
            proposed_boxes: Tensor with shape (num_proposals, 5).
            gt_boxes_per_class: Tensor holding the ground truth boxes for each
                class. Has shape (num_classes, num_gt_boxes_per_class, 4).

        Returns:
            A tensor with shape (num_proposals, num_classes * 4), holding the
            correct bbox_preds.
        """

        def bbox_encode(gt_boxes):
            return encode(
                proposed_boxes, gt_boxes
            )
        bbox_pred_tensor = tf.map_fn(
            bbox_encode, gt_boxes_per_class,
            dtype=tf.float32
        )
        # We need to explicitly unstack the tensor so that tf.concat works
        # properly.
        bbox_pred_list = tf.unstack(bbox_pred_tensor)
        return tf.concat(bbox_pred_list, 1)

    def _check_proposals_are_clipped(self, proposals, image_shape):
        """Asserts that no proposals exceed the image boundaries.
        """
        for proposal in proposals:
            self.assertLess(proposal[0], image_shape[1])
            self.assertLess(proposal[1], image_shape[0])
            self.assertLess(proposal[2], image_shape[1])
            self.assertLess(proposal[3], image_shape[0])
            for i in range(4):
                self.assertGreaterEqual(proposal[i], 0)

    def testBackgroundFilter(self):
        """Tests that we're not returning an object when a proposal is background.

        This includes two sub-tests. One case in which there is a foreground
        proposal, and one in which all proposals are background. We use the
        same proposed_boxes and gt_boxes, but change the cls_prob.
        """

        proposed_boxes = tf.constant([
            (85, 500, 730, 590),
            (50, 500, 70, 530),
            (700, 570, 740, 598),
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
        # Assertion for 'one foreground' case.
        # This assertion has two purposes:
        #     1. checking that we only get one object.
        #     2. checking that that object has the same box as class 2.
        # We take this to mean we're correctly ignoring the two proposals
        # where 'background' is the highest probability class.
        self.assertAllClose(
            proposal_prediction_one_foreground['objects'],
            self._compute_tf_graph(gt_boxes_per_class)[2],
            atol=self._equality_delta
        )
        # Assertion for 'all background' case.
        self.assertEqual(
            len(proposal_prediction_all_background['objects']), 0
        )

    def testNMSFilter(self):
        """Tests that we're applying NMS correctly.
        """

        # The first two boxes have a very high IoU between them. One of them
        # should be filtered by the NMS filter.
        proposed_boxes = tf.constant([
            (85, 500, 730, 590),
            (50, 500, 740, 570),
            (700, 570, 740, 598),
        ])
        gt_boxes_per_class = tf.constant([
            [(101, 101, 201, 249)],
            [(200, 502, 209, 532)],
            [(86, 571, 743, 599)],
        ])
        bbox_pred = self._get_bbox_pred(proposed_boxes, gt_boxes_per_class)
        cls_prob = tf.constant([
            (0., .1, .3, .6),
            (.1, .2, .25, .45),
            (.2, .3, .25, .25),
        ])

        proposal_prediction = self._run_rcnn_proposal(
            self._shared_model,
            proposed_boxes,
            bbox_pred,
            cls_prob,
        )
        labels = proposal_prediction['proposal_label']

        # Assertions
        self.assertEqual(proposal_prediction['objects'].shape[0], 2)
        self.assertIn(0, labels)
        self.assertIn(2, labels)

    def testImageClipping(self):
        """Tests that we're clipping images correctly.

        We test two image shapes, (1440, 900) and (900, 1440). Note we pass
        shapes as (height, width).
        """

        proposed_boxes = tf.constant([
            (1300, 800, 1435, 870),
            (10, 1, 30, 7),
            (2, 870, 80, 898),
        ])
        gt_boxes_per_class = tf.constant([
            [(1320, 815, 1455, 912)],
            [(5, -8, 31, 8)],
            [(-120, 910, 78, 1040)],
        ])
        bbox_pred = self._get_bbox_pred(proposed_boxes, gt_boxes_per_class)
        cls_prob = tf.constant([
            (0., 1., 0., 0.),
            (.2, .25, .3, .25),
            (.45, 0., 0., .55),
        ])

        shape1 = (1440, 900)
        shape2 = (900, 1440)

        proposal_prediction_shape1 = self._run_rcnn_proposal(
            self._shared_model,
            proposed_boxes,
            bbox_pred,
            cls_prob,
            image_shape=shape1,
        )
        proposal_prediction_shape2 = self._run_rcnn_proposal(
            self._shared_model,
            proposed_boxes,
            bbox_pred,
            cls_prob,
            image_shape=shape2,
        )
        # Assertions
        self._check_proposals_are_clipped(
            proposal_prediction_shape1['objects'],
            shape1,
        )
        self._check_proposals_are_clipped(
            proposal_prediction_shape2['objects'],
            shape2,
        )

    def testBboxPred(self):
        """Tests that we're using bbox_pred correctly.
        """

        proposed_boxes = tf.constant([
            (200, 315, 400, 370),
            (56, 0, 106, 4),
            (15, 15, 20, 20),
        ])

        gt_boxes_per_class = tf.constant([
            [(0, 0, 1, 1)],
            [(5, 5, 10, 10)],
            [(15, 15, 20, 20)],
        ])
        bbox_pred = self._get_bbox_pred(proposed_boxes, gt_boxes_per_class)

        cls_prob = tf.constant([
            (0., 1., 0., 0.),
            (.2, .25, .3, .25),
            (.45, 0., 0., .55),
        ])

        proposal_prediction = self._run_rcnn_proposal(
            self._shared_model,
            proposed_boxes,
            bbox_pred,
            cls_prob,
        )

        objects = self._compute_tf_graph(
            tf.squeeze(gt_boxes_per_class, axis=1)
        )
        # We need to sort the objects by `cls_prob` from high to low score.
        cls_prob = self._compute_tf_graph(cls_prob)
        # Ignoring background prob get the reverse argsort for the max of each
        # object.
        decreasing_idxs = cls_prob[:, 1:].max(axis=1).argsort()[::-1]
        # Sort by indexing.
        objects_sorted = objects[decreasing_idxs]

        self.assertAllClose(
            proposal_prediction['objects'],
            objects_sorted,
            atol=self._equality_delta
        )

    def testLimits(self):
        """Tests that we're respecting the limits imposed by the config.
        """

        limits_config = self._config.copy()
        limits_config['class_max_detections'] = 2
        limits_config['total_max_detections'] = 3
        limits_config = EasyDict(limits_config)
        limits_num_classes = 2
        limits_model = RCNNProposal(limits_num_classes, limits_config)

        proposed_boxes = tf.constant([
            (0, 0, 1, 1),  # class 0
            (5, 5, 10, 10),  # class 1
            (15, 15, 20, 20),  # class 1
            (25, 25, 30, 30),  # class 0
            (35, 35, 40, 40),
            (38, 40, 65, 65),
            (70, 50, 90, 90),  # class 0
            (95, 95, 100, 100),
            (105, 105, 110, 110),  # class 1
        ])
        # All zeroes for our bbox_pred.
        bbox_pred = tf.constant([[0.] * limits_num_classes * 4] * 9)
        cls_prob = tf.constant([
            (0., 1., 0.),
            (0., .2, .8),
            (0., .45, .55),
            (0., .55, .45),
            (1., 0., 0.),
            (1., 0., 0.),
            (0., .95, .05),
            (1., 0., 0.),
            (0., .495, .505),
        ])

        proposal_prediction = self._run_rcnn_proposal(
            limits_model,
            proposed_boxes,
            bbox_pred,
            cls_prob,
        )
        labels = proposal_prediction['proposal_label']
        num_class0 = labels[labels == 0].shape[0]
        num_class1 = labels[labels == 1].shape[0]

        self.assertLessEqual(num_class0, limits_config.class_max_detections)
        self.assertLessEqual(num_class1, limits_config.class_max_detections)
        num_total = labels.shape[0]
        self.assertLessEqual(num_total, limits_config.total_max_detections)


if __name__ == '__main__':
    tf.test.main()
