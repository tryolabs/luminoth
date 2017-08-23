import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn.rpn_target import RPNTarget


class RPNTargetTest(tf.test.TestCase):

    def setUp(self):
        super(RPNTargetTest, self).setUp()
        # Setup
        self.gt_boxes = np.array([[200, 0, 400, 400]])
        self.im_size = (600, 600)
        self.config = EasyDict({
            'allowed_border': 0,
            'clobber_positives': False,
            'foreground_threshold': 0.7,
            'background_threshold_high': 0.3,
            'foreground_fraction': 0.5,
            'minibatch_size': 2
        })

    def _run_rpn_target(self, anchors, config):
        gt_boxes = tf.placeholder(tf.float32, shape=self.gt_boxes.shape)
        im_size = tf.placeholder(tf.float32, shape=(2,))
        all_anchors = tf.placeholder(tf.float32, shape=anchors.shape)

        model = RPNTarget(anchors.shape[0], config, debug=True)
        labels, bbox_targets, max_overlaps = model(
            all_anchors, gt_boxes, im_size)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            labels_val, bbox_targets_val, max_overlaps_val = sess.run(
                [labels, bbox_targets, max_overlaps], feed_dict={
                    gt_boxes: self.gt_boxes,
                    im_size: self.im_size,
                    all_anchors: anchors,
                })
            return labels_val, bbox_targets_val, max_overlaps_val

    def testBaseCase(self):
        """
        Tests a basic case that includes foreground and backgrounds
        """
        all_anchors = np.array([
            [200, 100, 400, 400],  # foreground
            [300, 300, 400, 400],  # background
            [200, 380, 300, 500],  # background
        ], dtype=np.float32)
        labels, bbox_targets, max_overlaps = self._run_rpn_target(
            all_anchors, self.config
        )

        # Check we get exactly a foreground, an ignored background (because of
        # minibatch_size = 2) and a background. Also checks they are in the
        # correct order
        self.assertAllEqual(
            labels,
            np.array([1, 0, -1])
        )

        # Check max_overlaps shape
        self.assertEqual(
            max_overlaps.shape,
            (3,)
        )

        # Check that the foreground has overlap > 0.7.
        # Only the first value is checked because it's the only anchor assigned
        # to foreground.
        self.assertGreaterEqual(
            max_overlaps[0],
            0.7
        )

        # Check that the backgrounds have overlaps < 0.3.
        self.assertEqual(
            np.less_equal(max_overlaps[1:], 0.3).all(),
            True
        )

    def testBorderOutsiders(self):
        """
        Test with anchors that fall partially outside the image.
        """
        all_anchors = np.array([
            [200, 100, 400, 400],  # foreground
            [300, 300, 400, 400],  # background
            [200, 380, 300, 500],  # background
            [500, 500, 600, 650],  # border outsider
            [200, 100, 400, 400],  # foreground
        ])

        config = self.config
        config['minibatch_size'] = 5
        labels, bbox_targets, max_overlaps = self._run_rpn_target(
            all_anchors, config
        )

        # Checks that the order of labels for anchors is correct.
        # The fourth anchor is always ignored because it has a point outside
        # the image.
        self.assertAllEqual(
            labels,
            np.array([1, 0, 0, -1, 1])
        )

        # Test with a different foreground_fraction value.
        config['foreground_fraction'] = 0.2
        labels, bbox_targets, max_overlaps = self._run_rpn_target(
            all_anchors, config
        )

        # Checks that the order of labels for anchors is correct.
        # The fourth anchor is always ignored because it has a point outside
        # the image.
        self.assertAllEqual(
            labels,
            np.array([1, 0, 0, -1, -1])
        )

    def testWithNoClearMatch(self):
        """
        Tests that despite a foreground doesn't exist, an anchor is always
        assigned.
        Also tests the positive clobbering behaviour setting.
        """
        all_anchors = np.array([
            [300, 300, 400, 400],  # background
            [200, 380, 300, 500],  # background
        ])

        labels, bbox_targets, max_overlaps = self._run_rpn_target(
            all_anchors, self.config
        )

        # Check we get only one foreground anchor.
        self.assertAllEqual(
            labels,
            np.array([1, 0])
        )

        config = self.config
        config['clobber_positives'] = True

        labels, bbox_targets, max_overlaps = self._run_rpn_target(
            all_anchors, config
        )

        # Check we don't get a foreground anchor because of possitive
        # clobbering enabled.
        self.assertAllEqual(
            labels,
            np.array([0, 0])
        )


if __name__ == "__main__":
    tf.test.main()
