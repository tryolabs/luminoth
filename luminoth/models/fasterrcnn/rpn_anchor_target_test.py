import sonnet as snt
import tensorflow as tf
import numpy as np

from .rpn_anchor_target import RPNAnchorTarget
from luminoth.utils.generate_anchors import generate_anchors


class RPNAnchorTargetTest(tf.test.TestCase):

    def setUp(self):
        super(RPNAnchorTargetTest, self).setUp()
        # Setup anchors
        self.anchor_scales = np.array([8, 16, 32])
        self.anchor_ratios = np.array([0.5, 1, 2])
        self.anchors = generate_anchors(
            ratios=self.anchor_ratios, scales=self.anchor_scales)

    def testBasic(self):
        model = RPNAnchorTarget(self.anchors)
        rpn_cls_score_shape = (1, 32, 32, model._num_anchors * 2)
        gt_boxes_shape = (1, 4)  # 1 ground truth boxes.
        im_info_shape = (2,)

        rpn_cls_score_ph = tf.placeholder(
            tf.float32, shape=rpn_cls_score_shape)
        gt_boxes_ph = tf.placeholder(tf.float32, shape=gt_boxes_shape)
        im_info_ph = tf.placeholder(tf.float32, shape=im_info_shape)

        out = model(rpn_cls_score_ph, gt_boxes_ph, im_info_ph)

        # TODO: Can't test with random values. Needs some compatible values.
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_inst = sess.run(out, feed_dict={
                rpn_cls_score_ph: np.random.rand(*rpn_cls_score_shape),
                gt_boxes_ph: [[ 45,  42, 455, 342]],
                im_info_ph: [375, 500],
            })


if __name__ == "__main__":
    tf.test.main()
