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
            proposed_boxes: Tensor with shape (num_proposals, 4).
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
        """Asserts that no proposals exceed the image boundaries."""
        for proposal in proposals:
            self.assertLess(proposal[0], image_shape[1])
            self.assertLess(proposal[1], image_shape[0])
            self.assertLess(proposal[2], image_shape[1])
            self.assertLess(proposal[3], image_shape[0])
            for i in range(4):
                self.assertGreaterEqual(proposal[i], 0)

    def testNoBackgroundClass(self):
        """Tests that we're not returning an object with background class.

        For this, we make sure all predictions have a class between 0 and 2.
        That is, even though we have four classes (three plus background),
        background is completely ignored.
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

        # We build one prediction for each class.
        cls_prob = tf.constant([
            (0., .3, .3, .4),
            (.8, 0., 0., 2.),
            (.35, .3, .2, .15),
        ])

        proposal_prediction = self._run_rcnn_proposal(
            self._shared_model,
            proposed_boxes,
            bbox_pred,
            cls_prob,
        )

        # Make sure we get 3 predictions, one per class (as they're NMSed int
        # a single one).
        self.assertEqual(len(proposal_prediction['objects']), 3)
        self.assertIn(0, proposal_prediction['proposal_label'])
        self.assertIn(1, proposal_prediction['proposal_label'])
        self.assertIn(2, proposal_prediction['proposal_label'])

    def testNMSFilter(self):
        """Tests that we're applying NMS correctly."""

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

        # All proposals are mapped perfectly into each GT box, so we should
        # have 3 resulting objects after applying NMS.
        self.assertEqual(len(proposal_prediction['objects']), 3)

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
        """Tests that we're using bbox_pred correctly."""

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
        """Tests that we're respecting the limits imposed by the config."""

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
