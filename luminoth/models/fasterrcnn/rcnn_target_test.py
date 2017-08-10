import tensorflow as tf
import numpy as np

from easydict import EasyDict
from luminoth.models.fasterrcnn.rcnn_target import RCNNTarget


class RCNNTargetTest(tf.test.TestCase):

    def setUp(self):
        super(RCNNTargetTest, self).setUp()

        # We don't care about the class labels or the batch number in most of these tests.
        self._num_classes = 5
        self._placeholder_label = 3.
        self._batch_number = 1

        self._image_size = (800, 600)

        self._config = EasyDict({
            'allowed_border': 0,
            'clobber_positives': False,
            'foreground_threshold': 0.7,
            'background_threshold_high': 0.5,
            'background_threshold_low': 0.1,
            'foreground_fraction': 0.5,
            'minibatch_size': 2,
        })
        # We check for a difference smaller than this numbers in our tests instead of checking
        # for exact equality.
        self._equality_delta = 1e-03

        self._shared_model = RCNNTarget(self._num_classes, self._config)

    def _run_rcnn_target(self, model, gt_boxes, proposed_boxes):
        """Runs an instance of RCNNTarget

        Arguments:
            model: an RCNNTarget model.
            gt_boxes: a Tensor holding the ground truth boxes, with shape (num_gt, 5). The last
                value is the class label.
            proposed_boxes: a Tensor holding the proposed boxes. Its shape is (num_proposals, 5).
                The first value is the batch number.

        Returns:
            The tuple returned by RCNNTarget._build().
        """
        rcnn_target_net = model(proposed_boxes, gt_boxes)
        with self.test_session() as sess:
            return sess.run(rcnn_target_net)

    def testBasic(self):
        """Tests a basic case.

        We have one ground truth box and three proposals. One should be background, one foreground,
        and one should be an ignored background (i.e. less IoU than whatever value is set as
        config.background_threshold_low).
        """

        gt_boxes = tf.constant([(20, 20, 80, 100, self._placeholder_label)])

        proposed_boxes = tf.constant([
            (self._batch_number, 55, 75, 85, 105),  # Background box, IoU ~0.1293
            (self._batch_number, 25, 21, 85, 105),  # Foreground box, IoU ~0.7934
            (self._batch_number, 78, 98, 99, 135),  # Ignored box, IoU ~0.0015
        ])

        (proposals_label, bbox_targets) = self._run_rcnn_target(self._shared_model,
                                                                gt_boxes,
                                                                proposed_boxes)

        # We test that all values are 'close' (up to self._equality_delta) instead of equal to
        # avoid failing due to a floating point rounding error.
        # We sum 1 to the placeholder label because rcnn_target does the same due to the fact that
        # it uses 0 to signal 'background'.
        self.assertAllClose(proposals_label, np.array([0., self._placeholder_label + 1, -1.]),
                            atol=self._equality_delta)

    def testEmptyCase(self):
        """Tests that we're choosing the best box when none are above the foreground threshold.
        """

        gt_boxes = tf.constant([(423, 30, 501, 80, self._placeholder_label)])

        proposed_boxes = tf.constant([
            (self._batch_number, 491, 70, 510, 92),  # IoU 0.0277
            (self._batch_number, 400, 60, 450, 92),  # IoU 0.1147
            (self._batch_number, 413, 40, 480, 77),  # IoU 0.4998: highest
            (self._batch_number, 411, 40, 480, 77),  # IoU 0.4914
        ])

        (proposals_label, bbox_targets) = self._run_rcnn_target(self._shared_model,
                                                                gt_boxes,
                                                                proposed_boxes)
        # Assertions
        self.assertAlmostEqual(proposals_label[2], self._placeholder_label + 1,
                               delta=self._equality_delta)
        for i, label in enumerate(proposals_label):
            if i != 2:
                self.assertLess(label, 1)

    def testMultipleOverlap(self):
        """Tests that we're choosing the best box when several are foreground for the same gt box.
        """

        gt_boxes = tf.constant([(200, 300, 250, 390, self._placeholder_label)])

        proposed_boxes = tf.constant([
            (self._batch_number, 12, 70, 350, 540),  # noise
            (self._batch_number, 0, 0, 400, 400),  # noise
            (self._batch_number, 197, 300, 252, 389),  # IoU: 0.9015: highest
            (self._batch_number, 196, 300, 252, 389),  # IoU: 0.8859
            (self._batch_number, 197, 303, 252, 394),  # IoU: 0.8459
            (self._batch_number, 197, 302, 252, 389),  # IoU: 0.8832
        ])

        (proposals_label, bbox_targets) = self._run_rcnn_target(self._shared_model,
                                                                gt_boxes,
                                                                proposed_boxes)
        # Assertions
        self.assertAlmostEqual(proposals_label[2], self._placeholder_label + 1,
                               delta=self._equality_delta)
        for i, label in enumerate(proposals_label):
            if i != 2:
                self.assertLess(label, 1)

if __name__ == '__main__':
    tf.test.main()
