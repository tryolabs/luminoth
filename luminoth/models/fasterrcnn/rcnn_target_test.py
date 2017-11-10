import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn.rcnn_target import RCNNTarget
from luminoth.utils.test.gt_boxes import generate_gt_boxes


class RCNNTargetTest(tf.test.TestCase):

    def setUp(self):
        super(RCNNTargetTest, self).setUp()

        # We don't care about the class labels or the batch number in most of
        # these tests.
        self._num_classes = 5
        self._placeholder_label = 3.

        self._config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            'background_threshold_low': 0.1,
            'foreground_fraction': 0.5,
            'minibatch_size': 2,
        })
        # We check for a difference smaller than this numbers in our tests
        # instead of checking for exact equality.
        self._equality_delta = 1e-03

        self._shared_model = RCNNTarget(
            self._num_classes, self._config, seed=0
        )
        tf.reset_default_graph()

    def _run_rcnn_target(self, model, gt_boxes, proposed_boxes):
        """Runs an instance of RCNNTarget

        Args:
            model: an RCNNTarget model.
            gt_boxes: a Tensor holding the ground truth boxes, with shape
                (num_gt, 5). The last value is the class label.
            proposed_boxes: a Tensor holding the proposed boxes. Its shape is
                (num_proposals, 4). The first value is the batch number.

        Returns:
            The tuple returned by RCNNTarget._build().
        """
        rcnn_target_net = model(proposed_boxes, gt_boxes)
        with self.test_session() as sess:
            return sess.run(rcnn_target_net)

    def testBasic(self):
        """Tests a basic case.

        We have one ground truth box and three proposals. One should be
        background, one foreground, and one should be an ignored background
        (i.e. less IoU than whatever value is set as
        config.background_threshold_low).
        """

        gt_boxes = tf.constant(
            [(20, 20, 80, 100, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (55, 75, 85, 105),  # Background
                                    # IoU ~0.1293
                (25, 21, 85, 105),  # Foreground
                                    # IoU ~0.7934
                (78, 98, 99, 135),  # Ignored
                                    # IoU ~0.0015
            ],
            dtype=tf.float32
        )

        proposals_label, bbox_targets = self._run_rcnn_target(
            self._shared_model, gt_boxes, proposed_boxes
        )
        # We test that all values are 'close' (up to self._equality_delta)
        # instead of equal to avoid failing due to a floating point rounding
        # error.
        # We sum 1 to the placeholder label because rcnn_target does the same
        # due to the fact that it uses 0 to signal 'background'.
        self.assertAllClose(
            proposals_label,
            np.array([0., self._placeholder_label + 1, -1.]),
            atol=self._equality_delta
        )

        self.assertEqual(
            proposals_label[proposals_label >= 0].shape[0],
            self._config.minibatch_size
        )

    def testEmptyCase(self):
        """Tests we're choosing the best box when none are above the
        foreground threshold.
        """

        gt_boxes = tf.constant(
            [(423, 30, 501, 80, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (491, 70, 510, 92),  # IoU 0.0277
                (400, 60, 450, 92),  # IoU 0.1147
                (413, 40, 480, 77),  # IoU 0.4998: highest
                (411, 40, 480, 77),  # IoU 0.4914
            ],
            dtype=tf.float32
        )

        proposals_label, bbox_targets = self._run_rcnn_target(
            self._shared_model, gt_boxes, proposed_boxes
        )

        # Assertions
        self.assertAlmostEqual(
            proposals_label[2], self._placeholder_label + 1,
            delta=self._equality_delta
        )

        for i, label in enumerate(proposals_label):
            if i != 2:
                self.assertLess(label, 1)

        self.assertEqual(
            proposals_label[proposals_label >= 0].shape[0],
            self._config.minibatch_size
        )

    def testAbsolutelyEmptyCase(self):
        """Tests the code doesn't break when there's no proposals with IoU > 0.
        """

        gt_boxes = tf.constant(
            [(40, 90, 100, 105, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (0, 0, 39, 89),
                (101, 106, 300, 450),
                (340, 199, 410, 420),
            ],
            dtype=tf.float32
        )

        (proposals_label, bbox_targets) = self._run_rcnn_target(
            self._shared_model,
            gt_boxes,
            proposed_boxes
        )
        foreground_fraction = self._config.foreground_fraction
        minibatch_size = self._config.minibatch_size

        foreground_number = proposals_label[proposals_label >= 1].shape[0]
        background_number = proposals_label[proposals_label == 0].shape[0]

        self.assertGreater(foreground_number, 0)
        self.assertLessEqual(
            foreground_number,
            np.floor(foreground_fraction * minibatch_size)
        )
        self.assertLessEqual(
            background_number,
            minibatch_size - foreground_number
        )

    def testMultipleOverlap(self):
        """Tests that we're choosing a foreground box when there's several for
        the same gt box.
        """

        gt_boxes = tf.constant(
            [(200, 300, 250, 390, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (12, 70, 350, 540),  # noise
                (190, 310, 240, 370),  # IoU: 0.4763
                (197, 300, 252, 389),  # IoU: 0.9015
                (196, 300, 252, 389),  # IoU: 0.8859
                (197, 303, 252, 394),  # IoU: 0.8459
                (180, 310, 235, 370),  # IoU: 0.3747
                (0, 0, 400, 400),  # noise
                (197, 302, 252, 389),  # IoU: 0.8832
                (0, 0, 400, 400),  # noise
            ],
            dtype=tf.float32
        )

        proposals_label, bbox_targets = self._run_rcnn_target(
            self._shared_model, gt_boxes, proposed_boxes
        )

        # Assertions
        foreground_number = proposals_label[proposals_label >= 1].shape[0]
        background_number = proposals_label[proposals_label == 0].shape[0]

        foreground_fraction = self._config.foreground_fraction

        self.assertEqual(
            foreground_number,
            np.floor(self._config.minibatch_size * foreground_fraction),
        )
        self.assertEqual(
            background_number,
            self._config.minibatch_size - foreground_number,
        )

        foreground_idxs = np.nonzero(proposals_label >= 1)
        for foreground_idx in foreground_idxs:
            self.assertIn(foreground_idx, [2, 3, 4, 7])

        self.assertEqual(
            proposals_label[proposals_label >= 0].shape[0],
            self._config.minibatch_size
        )

    def testOddMinibatchSize(self):
        """Tests we're getting the right results when there's an odd minibatch
        size.
        """

        config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            'background_threshold_low': 0.1,
            'foreground_fraction': 0.5,
            'minibatch_size': 5,
        })

        model = RCNNTarget(self._num_classes, config, seed=0)

        gt_boxes = tf.constant(
            [(200, 300, 250, 390, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (12, 70, 350, 540),  # noise
                (190, 310, 240, 370),  # IoU: 0.4763
                (197, 300, 252, 389),  # IoU: 0.9015
                (196, 300, 252, 389),  # IoU: 0.8859
                (197, 303, 252, 394),  # IoU: 0.8459
                (180, 310, 235, 370),  # IoU: 0.3747
                (0, 0, 400, 400),  # noise
                (197, 302, 252, 389),  # IoU: 0.8832
                (180, 310, 235, 370),  # IoU: 0.3747
                (180, 310, 235, 370),  # IoU: 0.3747
                (0, 0, 400, 400),  # noise
            ],
            dtype=tf.float32
        )

        (proposals_label, bbox_targets) = self._run_rcnn_target(
            model,
            gt_boxes,
            proposed_boxes
        )

        foreground_number = proposals_label[proposals_label >= 1].shape[0]
        background_number = proposals_label[proposals_label == 0].shape[0]

        foreground_fraction = config.foreground_fraction
        minibatch_size = config.minibatch_size

        self.assertLessEqual(
            foreground_number,
            np.floor(foreground_fraction * minibatch_size)
        )
        self.assertGreater(foreground_number, 0)
        self.assertLessEqual(
            background_number,
            minibatch_size - foreground_number
        )

        self.assertEqual(
            proposals_label[proposals_label >= 0].shape[0],
            config.minibatch_size
        )

    def testBboxTargetConsistency(self):
        """Tests that bbox_targets is consistent with proposals_label.

        That means we test that we have the same number of elements in
        bbox_targets and proposals_label, and that only the proposals
        marked with a class are assigned a non-zero bbox_target.
        """

        config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            'background_threshold_low': 0,  # use 0 to get complete batch
            'foreground_fraction': 0.5,
            # We change the minibatch_size the catch all our foregrounds
            'minibatch_size': 8,
        })

        model = RCNNTarget(self._num_classes, config, seed=0)

        gt_boxes = tf.constant(
            [(200, 300, 250, 390, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (12, 70, 350, 540),  # noise
                (190, 310, 240, 370),  # IoU: 0.4763
                (197, 300, 252, 389),  # IoU: 0.9015
                (196, 300, 252, 389),  # IoU: 0.8859
                (197, 303, 252, 394),  # IoU: 0.8459
                (180, 310, 235, 370),  # IoU: 0.3747
                (0, 0, 400, 400),  # noise
                (197, 302, 252, 389),  # IoU: 0.8832
                (0, 0, 400, 400),  # noise
            ],
            dtype=tf.float32
        )

        (proposals_label, bbox_targets) = self._run_rcnn_target(
            model,
            gt_boxes,
            proposed_boxes
        )

        foreground_idxs = np.nonzero(proposals_label >= 1)
        non_empty_bbox_target_idxs = np.nonzero(np.any(bbox_targets, axis=1))

        self.assertAllEqual(
            foreground_idxs, non_empty_bbox_target_idxs
        )
        self.assertGreater(proposals_label[proposals_label >= 1].shape[0], 0)
        self.assertEqual(
            proposals_label[proposals_label >= 0].shape[0],
            config.minibatch_size
        )

    def testMultipleGtBoxes(self):
        """Tests we're getting the right labels when there's several gt_boxes.
        """

        num_classes = 3
        config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            'background_threshold_low': 0.1,
            'foreground_fraction': 0.5,
            # We change the minibatch_size the catch all our foregrounds
            'minibatch_size': 18,
        })
        model = RCNNTarget(num_classes, config, seed=0)

        gt_boxes = tf.constant(
            [
                (10, 0, 398, 399, 0),
                (200, 300, 250, 390, 1),
                (185, 305, 235, 372, 2),
            ],
            dtype=tf.float32
        )
        proposed_boxes = tf.constant(
            [
                (12, 70, 350, 540),  # noise
                (190, 310, 240, 370),  # 2
                (197, 300, 252, 389),  # 1
                (196, 300, 252, 389),  # 1
                (197, 303, 252, 394),  # 1
                (180, 310, 235, 370),  # 2
                (0, 0, 400, 400),  # 0
                (197, 302, 252, 389),  # 1
                (0, 0, 400, 400),  # 0
            ],
            dtype=tf.float32
        )

        (proposals_label, bbox_targets) = self._run_rcnn_target(
            model,
            gt_boxes,
            proposed_boxes
        )
        # We don't care much about the first value.
        self.assertAllClose(
            proposals_label[1:],
            # We sum one to normalize for RCNNTarget's results.
            np.add([2., 1., 1., 1., 2., 0., 1., 0.], 1),
            self._equality_delta
        )

    def testNonZeroForegrounds(self):
        """Tests we never get zero foregrounds.

        We're doing iterations with random gt_boxes and proposals under
        conditions that make it likely we would get zero foregrounds if there
        is a bug in the code. (Very few gt_boxes and a small minibatch_size).
        """
        number_of_iterations = 50
        for _ in range(number_of_iterations):
            im_shape = np.random.randint(
                low=600, high=980, size=2, dtype=np.int32
            )
            total_boxes = np.random.randint(
                low=1, high=4, dtype=np.int32
            )
            total_proposals = np.random.randint(
                low=4, high=8, dtype=np.int32
            )
            # Generate gt_boxes and then add a label.
            gt_boxes = generate_gt_boxes(
                total_boxes, im_shape
            )
            gt_boxes_w_label = np.concatenate(
                [gt_boxes, [[self._placeholder_label]] * total_boxes],
                axis=1
            ).astype(np.float32)
            # Generate the proposals and add the batch number.
            proposed_boxes = generate_gt_boxes(
                total_proposals, im_shape
            )
            # Run RCNNTarget.
            (proposals_label, _) = self._run_rcnn_target(
                self._shared_model, gt_boxes_w_label,
                proposed_boxes.astype(np.float32)
            )
            # Assertion
            foreground_number = proposals_label[proposals_label > 0].shape[0]
            self.assertGreater(foreground_number, 0)

    def testCorrectBatchSize(self):
        config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            # Use zero to get all non matching as backgrounds.
            'background_threshold_low': 0.0,
            'foreground_fraction': 0.5,
            # We change the minibatch_size the catch all our foregrounds
            'minibatch_size': 64,
        })

        gt_boxes = tf.constant([
            [10, 10, 20, 20, 0]
        ], dtype=tf.float32)

        proposed_boxes_backgrounds = [[21, 21, 30, 30]] * 100
        proposed_boxes_foreground = [[11, 11, 19, 19]] * 100

        proposed_boxes = tf.constant(
            proposed_boxes_backgrounds + proposed_boxes_foreground,
            dtype=tf.float32
        )

        model = RCNNTarget(self._num_classes, config, seed=0)

        (proposals_label, bbox_targets) = self._run_rcnn_target(
            model,
            gt_boxes,
            proposed_boxes
        )

        self.assertEqual(
            proposals_label[proposals_label >= 0].shape[0],
            config.minibatch_size
        )

    def testLabelPriority(self):
        """Tests we're prioritizing being the best proposal for a gt_box in
        label selection.
        """

        first_label = self._placeholder_label
        second_label = self._placeholder_label + 1

        num_classes = second_label + 10

        # We need a custom config to have a larger minibatch_size.
        config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            # Use zero to get all non matching as backgrounds.
            'background_threshold_low': 0.0,
            'foreground_fraction': 0.5,
            # We change the minibatch_size the catch all our foregrounds
            'minibatch_size': 64,
        })
        model = RCNNTarget(num_classes, config, seed=0)

        gt_boxes = tf.constant([
            [10, 10, 20, 20, first_label],
            [10, 10, 30, 30, second_label]
        ], dtype=tf.float32)

        # Both proposals have the first gt_box as the best match, but one of
        # them should be assigned to the label of the second gt_box anyway.
        proposed_boxes = tf.constant([
            [10, 10, 20, 20],
            [12, 10, 20, 20],
        ], dtype=tf.float32)

        (proposals_label, _) = self._run_rcnn_target(
            model,
            gt_boxes,
            proposed_boxes
        )

        num_first_label = len(
            proposals_label[proposals_label == first_label + 1]
        )
        num_second_label = len(
            proposals_label[proposals_label == second_label + 1]
        )

        # Assertions
        self.assertEqual(num_first_label, 1)
        self.assertEqual(num_second_label, 1)


if __name__ == '__main__':
    tf.test.main()
