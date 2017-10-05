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
        tf.reset_default_graph()

    def _run_rpn_target(self, anchors, gt_boxes, config):
        gt_boxes_tf = tf.placeholder(tf.float32, shape=gt_boxes.shape)
        im_size = tf.placeholder(tf.float32, shape=(2,))
        all_anchors = tf.placeholder(tf.float32, shape=anchors.shape)

        model = RPNTarget(anchors.shape[0], config, seed=0)
        labels, bbox_targets, max_overlaps = model(
            all_anchors, gt_boxes_tf, im_size)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            labels_val, bbox_targets_val, max_overlaps_val = sess.run(
                [labels, bbox_targets, max_overlaps], feed_dict={
                    gt_boxes_tf: gt_boxes,
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
            all_anchors, self.gt_boxes, self.config
        )

        # Check we get exactly a foreground, an ignored background (because of
        # minibatch_size = 2) and a background. Also checks they are in the
        # correct order
        self.assertAllEqual(
            labels,
            np.array([1, 0, -1])
        )

        # Assert bbox targets we are ignoring are zero.
        self.assertAllEqual(
            bbox_targets[1:],
            [[0, 0, 0, 0], [0, 0, 0, 0]]
        )

        # Assert bbox target is not zero
        self.assertTrue((bbox_targets[0] != 0).any())

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
            all_anchors, self.gt_boxes, config
        )

        # Checks that the order of labels for anchors is correct.
        # The fourth anchor is always ignored because it has a point outside
        # the image.
        self.assertAllEqual(
            labels,
            np.array([1, 0, 0, -1, 1])
        )

        # Check that foreground bbox targets are partially zero because
        # the X1 and width are already the same
        self.assertEqual(bbox_targets[0][0], 0)
        self.assertEqual(bbox_targets[0][2], 0)
        self.assertNotEqual(bbox_targets[0][1], 0)
        self.assertNotEqual(bbox_targets[0][3], 0)
        self.assertAllEqual(bbox_targets[0], bbox_targets[-1])

        # Assert bbox targets we are ignoring are zero.
        self.assertAllEqual(
            bbox_targets[1:4],
            np.zeros((3, 4))
        )

        # Test with a different foreground_fraction value.
        config['foreground_fraction'] = 0.2
        labels, bbox_targets, max_overlaps = self._run_rpn_target(
            all_anchors, self.gt_boxes, config
        )

        # Checks that the order of labels for anchors is correct.
        # The fourth anchor is always ignored because it has a point outside
        # the image.
        self.assertAllEqual(
            labels,
            np.array([1, 0, 0, -1, -1])
        )

        # Assert bbox targets we are ignoring are zero.
        self.assertAllEqual(
            bbox_targets[1:],
            np.zeros((4, 4))
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
            all_anchors, self.gt_boxes, self.config
        )

        # Check we get only one foreground anchor.
        self.assertAllEqual(
            labels,
            np.array([1, 0])
        )
        self.assertTrue((bbox_targets[0] != 0).all())
        self.assertAllEqual(bbox_targets[1], np.zeros(4))

        config = self.config
        config['clobber_positives'] = True

        labels, bbox_targets, max_overlaps = self._run_rpn_target(
            all_anchors, self.gt_boxes, config
        )

        # Check we don't get a foreground anchor because of possitive
        # clobbering enabled.
        self.assertAllEqual(
            labels,
            np.array([0, 0])
        )
        self.assertAllEqual(bbox_targets, np.zeros((2, 4)))

    def testWithMultipleGTBoxes(self):
        """
        Tests a basic case that includes two gt_boxes. What is going to happen
        is that for each gt_box its fixed at least a foreground. After if there
        are too many foreground, some of them will be disabled.
        """
        all_anchors = np.array([
            [300, 300, 400, 390],  # background IoU < 0.3
            [300, 300, 400, 400],  # foreground for the first gt_box
            [100, 310, 120, 380],  # foreground for the second gt_box
        ], dtype=np.float32)
        config = self.config
        config['minibatch_size'] = 3

        gt_boxes = np.array([[200, 0, 400, 400], [100, 300, 120, 375]])
        labels, bbox_targets, max_overlaps = self._run_rpn_target(
            all_anchors, gt_boxes, config
        )

        # Check we get exactly a foreground, in this case the first one,
        # a background and an ignored foreground.
        # Also checks they are in the correct order.
        self.assertAllEqual(
            labels,
            np.array([0, 1, -1])
        )

        # Assert bbox targets we are ignoring are zero.
        self.assertAllEqual(
            bbox_targets[0],
            [0, 0, 0, 0]
        )

        self.assertAllEqual(
             bbox_targets[2],
             [0, 0, 0, 0]
         )

        # Assert bbox target is not zero
        self.assertTrue((bbox_targets[1] != 0).any())

        # Check max_overlaps shape
        self.assertEqual(
            max_overlaps.shape,
            (3,)
        )

    def testWithManyGTBoxes(self):
        all_anchors = np.array([
            # Foregrounds
            [0, 0, 10, 10],
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [20, 20, 30, 30],
            [20, 20, 30, 30],
            [30, 30, 40, 40],
            [30, 30, 40, 40],
            # Backgrounds
            [100, 100, 110, 110],
            [100, 100, 120, 120],
            [110, 110, 120, 120],
            [110, 110, 130, 130],
            [110, 110, 120, 120],
            [110, 110, 130, 130],
            [110, 110, 120, 120],
            [110, 110, 130, 130],
        ], dtype=np.float32)
        config = self.config
        config['minibatch_size'] = 8  # 4 foregrounds and 4 backgrounds

        gt_boxes = np.array([
            [2, 2, 8, 8], [12, 12, 18, 18], [22, 22, 28, 28], [32, 32, 38, 38]
        ])
        labels, bbox_targets, max_overlaps = self._run_rpn_target(
            all_anchors, gt_boxes, config
        )

        # 4 foreground
        self.assertEqual(labels[labels == 1].shape[0], 4)
        # 4 background
        self.assertEqual(labels[labels == 0].shape[0], 4)

        # Check all 4 foregrounds are in the first 8 places of the label.
        # TODO: Ideally it should be one for each GT.
        self.assertTrue((labels.argsort()[-4:] < 8).all())


if __name__ == "__main__":
    tf.test.main()
