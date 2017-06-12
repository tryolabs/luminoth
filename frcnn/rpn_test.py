import sonnet as snt
import tensorflow as tf
import numpy as np

from sonnet.testing.parameterized import parameterized

from .rpn import RPN


class RPNTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

    def setUp(self):
        super(RPNTest, self).setUp()
        self.anchor_scales = [8, 16, 32]
        self.anchor_ratios = [0.5, 1, 2]
        self.num_channels = 512
        self.kernel_shape = [3, 3]

    def testConstructor(self):
        with self.assertRaisesRegexp(TypeError, 'anchor_scales must be iterable'):
            net = RPN(None, self.anchor_ratios)

        with self.assertRaisesRegexp(TypeError, 'anchor_ratios must be iterable'):
            net = RPN(self.anchor_scales, None)

        with self.assertRaisesRegexp(ValueError, 'anchor_scales must not be empty'):
            net = RPN([], self.anchor_ratios)

        with self.assertRaisesRegexp(ValueError, 'anchor_ratios must not be empty'):
            net = RPN(self.anchor_scales, [])

    def testBasic(self):
        model = RPN(
            self.anchor_scales, self.anchor_ratios, self.num_channels,
            self.kernel_shape
        )
        # With an image of (100, 100, 3) we get a VGG output of (32, 32, 512)
        # (plus the batch number)
        pretrained_output_shape = (1, 32, 32, 512)
        pretrained_output = tf.placeholder(
            tf.float32, shape=pretrained_output_shape)
        layers = model(pretrained_output)

        with self.test_session() as sess:
            # As in the case of a real session we need to initialize the
            # variables.
            sess.run(tf.global_variables_initializer())
            layers_inst = sess.run(layers, feed_dict={
                pretrained_output: np.random.rand(*pretrained_output_shape)
            })

        # Since pretrained
        rpn_shape = layers_inst['rpn'].shape
        # RPN has the same shape as the pretrained layer.
        self.assertEqual(pretrained_output_shape, rpn_shape)

        num_anchors = model._num_anchors

        # Class score generates 2 values per anchor
        rpn_cls_score_shape = layers_inst['rpn_cls_score'].shape
        rpn_cls_score_true_shape = pretrained_output_shape[
            :-1] + (num_anchors * 2,)  # num_anchors * 2
        self.assertEqual(rpn_cls_score_shape, rpn_cls_score_true_shape)

        rpn_cls_score_reshape_shape = layers_inst[
            'rpn_cls_score_reshape'].shape
        rpn_cls_score_reshape_true_shape = pretrained_output_shape[
            :-1] + (num_anchors * 2,)  # num_anchors * 2
        self.assertEqual(rpn_cls_score_reshape_shape, (1, 32, 288, 2))

        # RPN class prob shape has the spatial reshape
        rpn_cls_prob_shape = layers_inst['rpn_cls_prob'].shape
        self.assertEqual(rpn_cls_prob_shape, (1, 32, 288, 2))

        rpn_cls_prob_reshape_shape = layers_inst['rpn_cls_prob_reshape'].shape
        rpn_cls_prob_reshape_true_shape = pretrained_output_shape[
            :-1] + (num_anchors * 2,)  # num_anchors * 2
        self.assertEqual(rpn_cls_prob_reshape_shape,
                         rpn_cls_prob_reshape_true_shape)

        rpn_bbox_pred_shape = layers_inst['rpn_bbox_pred'].shape
        rpn_bbox_pred_true_shape = pretrained_output_shape[
            :-1] + (num_anchors * 4,)  # num_anchors * 4 (bbox regression)
        self.assertEqual(rpn_bbox_pred_shape, rpn_bbox_pred_true_shape)


if __name__ == "__main__":
    tf.test.main()
