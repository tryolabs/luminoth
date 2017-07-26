import tensorflow as tf
import numpy as np

from .rpn_proposal import RPNProposal
from .utils.generate_anchors import generate_anchors


class RPNProposalTest(tf.test.TestCase):
    def setUp(self):
        super(RPNProposalTest, self).setUp()
        # Setup anchors
        self.anchor_scales = np.array([8, 16, 32])
        self.anchor_ratios = np.array([0.5, 1, 2])
        self.anchors = generate_anchors(ratios=self.anchor_ratios, scales=self.anchor_scales)

    def testBasic(self):
        model = RPNProposal(self.anchors)
        rpn_cls_prob_shape = (1, 32, 32, model._num_anchors * 2)
        rpn_bbox_pred_shape = (1, 32, 32, model._num_anchors * 4)

        rpn_cls_prob_ph = tf.placeholder(tf.float32, shape=rpn_cls_prob_shape)
        rpn_bbox_pred_ph = tf.placeholder(tf.float32, shape=rpn_bbox_pred_shape)

        out = model(rpn_cls_prob_ph, rpn_bbox_pred_ph)

        # TODO: Can't test with random values. Needs some compatible values.
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_inst = sess.run(out, feed_dict={
                rpn_cls_prob_ph: np.random.rand(*rpn_cls_prob_shape),
                rpn_bbox_pred_ph: np.random.rand(*rpn_bbox_pred_shape),
            })


if __name__ == "__main__":
    tf.test.main()
