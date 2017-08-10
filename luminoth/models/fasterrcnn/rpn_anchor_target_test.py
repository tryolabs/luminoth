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
             [300, 300, 400, 400],  # background, must be ignored
             [200, 380, 300, 500],  # background
             ])
        self.config = easydict.EasyDict({
            'allowed_border': 0,
            'clobber_positives': 0,
            'foreground_threshold': 0.7,
            'background_threshold_high': 0.3,
            'foreground_fraction': 0.5,
            'minibatch_size': 2
        })
        self.pretrained_shape = (1, 1, 1, 1)

    def testBaseCase(self):
        """
        Tests a basic case that includes foreground and backgrounds
        """
        model = RPNAnchorTarget(self.all_anchors.shape[0], self.config)

        pretrained_shape = tf.placeholder(tf.float32, shape=(4,))
        gt_boxes = tf.placeholder(tf.float32, shape=self.gt_boxes.shape)
        im_size = tf.placeholder(tf.float32, shape=(2,))
        all_anchors = tf.placeholder(tf.float32, shape=self.all_anchors.shape)

        labels, bbox_targets, max_overlaps = model(
            pretrained_shape, gt_boxes, im_size, all_anchors
        )

        labels = tf.Print(labels, [labels, bbox_targets, max_overlaps])

        # Test
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            labels_val, bbox_targets_val, max_overlaps_val = sess.run(
                [labels, bbox_targets, max_overlaps], feed_dict={
                    pretrained_shape: self.pretrained_shape,
                    gt_boxes: self.gt_boxes,
                    im_size: self.im_size,
                    all_anchors: self.all_anchors,
                })

            # Check we get exactly and in order a foreground,
            # an ignored background (minibatch_size = 2) and a background.
            self.assertEqual(
                labels_val.all(),
                np.array([1, -1, 0]).all()
            )

            # Check that the foreground has overlaps > 0.7
            self.assertGreaterEqual(
                labels_val[0],
                0.7
            )

            # Check that the backgrounds have overlaps < 0.3
            self.assertLessEqual(
                labels_val[1:].all(),
                0.3
            )


if __name__ == "__main__":
    tf.test.main()
