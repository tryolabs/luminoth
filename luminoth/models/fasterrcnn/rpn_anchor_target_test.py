import tensorflow as tf
import numpy as np
import easydict

from rpn_anchor_target import RPNAnchorTarget


class RPNAnchorTargetTest(tf.test.TestCase):

    def setUp(self):
        super(RPNAnchorTargetTest, self).setUp()
        # Setup
        self.gt_boxes = np.array([[200, 0, 400, 400]])
        self.im_size = (600, 600)
        self.all_anchors = np.array(
            [[200, 100, 400, 400],  # foreground
             [300, 300, 400, 400],  # background
             [200, 380, 300, 500],  # background
             [500, 500, 600, 650],  # border outsider
             [200, 100, 400, 400],  # foreground
             ])
        self.config = easydict.EasyDict({
            'allowed_border': 0,
            'clobber_positives': False,
            'foreground_threshold': 0.7,
            'background_threshold_high': 0.3,
            'foreground_fraction': 0.5,
            'minibatch_size': 2
        })
        self.pretrained_shape = (1, 1, 1, 1)

    def execute(self, anchors, config):
        pretrained_shape = tf.placeholder(tf.float32, shape=(4,))
        gt_boxes = tf.placeholder(tf.float32, shape=self.gt_boxes.shape)
        im_size = tf.placeholder(tf.float32, shape=(2,))
        all_anchors = tf.placeholder(tf.float32, shape=anchors.shape)

        model = RPNAnchorTarget(anchors.shape[0], config)
        labels, bbox_targets, max_overlaps = model(
            pretrained_shape, gt_boxes, im_size, all_anchors
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            labels_val, bbox_targets_val, max_overlaps_val = sess.run(
                [labels, bbox_targets, max_overlaps], feed_dict={
                    pretrained_shape: self.pretrained_shape,
                    gt_boxes: self.gt_boxes,
                    im_size: self.im_size,
                    all_anchors: anchors,
                })
            return labels_val, bbox_targets_val, max_overlaps_val

    def testBaseCase(self):
        """
        Tests a basic case that includes foreground and backgrounds
        """
        labels_val, bbox_targets_val, max_overlaps_val = self.execute(
            self.all_anchors[:3], self.config)

        # Check we get exactly and in order a foreground,
        # an ignored background (minibatch_size = 2) and a background.
        self.assertAllEqual(
            labels_val,
            np.array([1, -1, 0])
        )

        # Check that the foreground has overlaps > 0.7.
        self.assertGreaterEqual(
            max_overlaps_val[0],
            0.7
        )

        # Check that the backgrounds have overlaps < 0.3.
        self.assertLessEqual(
            np.less_equal(max_overlaps_val[1:], 0.3).all(),
            True
        )

    def testNotEmpty(self):
        """
        Tests that despite doesn't exist a foreground, always an anchor is assigned
        """
        labels_val, bbox_targets_val, max_overlaps_val = self.execute(
            self.all_anchors[1:3], self.config)

        # Check we get an assigned anchor.
        self.assertAllEqual(
            labels_val,
            np.array([1, 0])
        )

    def testPositiveClobbering(self):
        """
        Tests the positive clobbering behaviour
        """
        config = self.config
        config['clobber_positives'] = True

        labels_val, bbox_targets_val, max_overlaps_val = self.execute(
            self.all_anchors[1:3], config)

        # Check we don't get an assigned anchor because of possitive clobbering.
        self.assertAllEqual(
            labels_val,
            np.array([0, 0])
        )

    def testBorderOutsiders(self):
        """
        Test with anchors outside the image
        """
        config = self.config
        config['minibatch_size'] = 4
        labels_val, bbox_targets_val, max_overlaps_val = self.execute(
            self.all_anchors[:5], config)

        # Check that the order or anchors is correct.
        self.assertAllEqual(
            labels_val,
            np.array([1, 0, 0, -1, 1])
        )


if __name__ == "__main__":
    tf.test.main()
