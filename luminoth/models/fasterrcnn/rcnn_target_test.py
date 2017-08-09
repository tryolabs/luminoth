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

    def testBasic(self):
        """Tests a basic case.

        We have one ground truth box and three proposals. One should be background, one foreground,
        and one should be an ignored background (i.e. less IoU than whatever value is set as
        config.background_threshold_low).
        """

        model = RCNNTarget(self._num_classes, self._config)

        gt_boxes = tf.constant([(20, 20, 80, 100, self._placeholder_label)])

        proposed_boxes = tf.constant([
            (self._batch_number, 55, 75, 85, 105),  # Background box
            (self._batch_number, 25, 21, 85, 105),  # Foreground box
            (self._batch_number, 78, 98, 99, 135),  # Ignored box
        ])

        rcnn_target_net = model(proposed_boxes, gt_boxes)

        proposals_label = []
        bbox_targets = []
        with self.test_session() as sess:
            (proposals_label, bbox_targets) = sess.run(rcnn_target_net)

        # We test that all values are 'close' (up to 1e-03 distance) to avoid failing due to a
        # floating point rounding error.
        # We sum 1 to the placeholder label because rcnn_target does the same due to the fact that
        # it uses 0 to signal 'background'.
        self.assertAllClose(proposals_label, np.array([0., self._placeholder_label + 1, -1.]),
                            atol=1e-03)


if __name__ == '__main__':
    tf.test.main()
