import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn.rpn import RPN
from luminoth.utils.anchors import generate_anchors_reference
from luminoth.utils.test import generate_gt_boxes, generate_anchors


class RPNTest(tf.test.TestCase):

    def setUp(self):
        super(RPNTest, self).setUp()
        self.num_anchors = 9
        # Use default settings.
        self.config = EasyDict({
            'num_channels': 512,
            'kernel_shape': [3, 3],
            'rpn_initializer': {
                'type': 'variance_scaling_initializer',
                'factor': 1.0,
                'mode': 'FAN_AVG',
                'uniform': True,
            },
            'cls_initializer': {
                'type': 'truncated_normal_initializer',
                'mean': 0.0,
                'stddev': 0.01,
            },
            'bbox_initializer': {
                'type': 'truncated_normal_initializer',
                'mean': 0.0,
                'stddev': 0.01,
            },
            'l2_regularization_scale': 0.0005,
            'l1_sigma': 3.0,
            'activation_function': 'relu6',
            'proposals': {
                'pre_nms_top_n': 12000,
                'post_nms_top_n': 2000,
                'nms_threshold': 0.6,
                'min_size': 0,
                'clip_after_nms': False,
                'filter_outside_anchors': False,
                'apply_nms': True,
                'min_prob_threshold': 0.0,
            },
            'target': {
                'allowed_border': 0,
                'clobber_positives': False,
                'foreground_threshold': 0.7,
                'background_threshold_high': 0.3,
                'background_threshold_low': 0.,
                'foreground_fraction': 0.5,
                'minibatch_size': 256,
            }
        })

        # Use default anchor configuration values.
        self.base_size = 256
        self.scales = np.array([0.5, 1, 2])
        self.ratios = np.array([0.5, 1, 2])
        self.stride = 16
        tf.reset_default_graph()

    def testBasic(self):
        """Tests shapes are consistent with anchor generation.
        """
        model = RPN(
            self.num_anchors, self.config, debug=True
        )
        # (plus the batch number)
        pretrained_output_shape = (1, 32, 32, 512)
        pretrained_output = tf.placeholder(
            tf.float32, shape=pretrained_output_shape)

        # Estimate image shape from the pretrained output and the anchor stride
        image_shape_val = (
            int(pretrained_output_shape[1] * self.stride),
            int(pretrained_output_shape[2] * self.stride),
        )

        # Use 4 ground truth boxes.
        gt_boxes_shape = (4, 4)
        gt_boxes = tf.placeholder(tf.float32, shape=gt_boxes_shape)
        image_shape_shape = (2,)
        image_shape = tf.placeholder(tf.float32, shape=image_shape_shape)
        # Total anchors depends on the pretrained output shape and the total
        # number of anchors per point.
        total_anchors = (
            pretrained_output_shape[1] * pretrained_output_shape[2] *
            self.num_anchors
        )
        all_anchors_shape = (total_anchors, 4)
        all_anchors = tf.placeholder(tf.float32, shape=all_anchors_shape)
        layers = model(
            pretrained_output, image_shape, all_anchors, gt_boxes=gt_boxes
        )

        with self.test_session() as sess:
            # As in the case of a real session we need to initialize the
            # variables.
            sess.run(tf.global_variables_initializer())
            layers_inst = sess.run(layers, feed_dict={
                # We don't really care about the value of the pretrained output
                # only that has the correct shape.
                pretrained_output: np.random.rand(
                    *pretrained_output_shape
                ),
                # Generate random but valid ground truth boxes.
                gt_boxes: generate_gt_boxes(
                    gt_boxes_shape[0], image_shape_val
                ),
                # Generate anchors from a reference and the shape of the
                # pretrained_output.
                all_anchors: generate_anchors(
                    generate_anchors_reference(
                        self.base_size, self.ratios, self.scales
                    ),
                    16,
                    pretrained_output_shape[1:3]
                ),
                image_shape: image_shape_val,
            })

        # Class score generates 2 values per anchor.
        rpn_cls_score_shape = layers_inst['rpn_cls_score'].shape
        rpn_cls_score_true_shape = (total_anchors, 2)
        self.assertEqual(rpn_cls_score_shape, rpn_cls_score_true_shape)

        # Probs have the same shape as cls scores.
        rpn_cls_prob_shape = layers_inst['rpn_cls_prob'].shape
        self.assertEqual(rpn_cls_prob_shape, rpn_cls_score_true_shape)

        # We check softmax with the sum of the output.
        rpn_cls_prob_sum = layers_inst['rpn_cls_prob'].sum(axis=1)
        self.assertAllClose(rpn_cls_prob_sum, np.ones(total_anchors))

        # Proposals and scores are related to the output of the NMS with
        # limits.
        total_proposals = layers_inst['proposals'].shape[0]
        total_scores = layers_inst['scores'].shape[0]

        # Check we don't get more than top_n proposals.
        self.assertGreaterEqual(
            self.config.proposals.post_nms_top_n, total_proposals
        )

        # Check we get a score for each proposal.
        self.assertEqual(total_proposals, total_scores)

        # Check that we get a regression for each anchor.
        self.assertEqual(
            layers_inst['rpn_bbox_pred'].shape,
            (total_anchors, 4)
        )

        # Check that we get a target for each regression for each anchor.
        self.assertEqual(
            layers_inst['rpn_bbox_target'].shape,
            (total_anchors, 4)
        )

        # Check that we get a target class for each anchor.
        self.assertEqual(
            layers_inst['rpn_cls_target'].shape,
            (total_anchors,)
        )

        # Check that targets are composed of [-1, 0, 1] only.
        rpn_cls_target = layers_inst['rpn_cls_target']
        self.assertEqual(
            tuple(np.sort(np.unique(rpn_cls_target))),
            (-1, 0., 1.)
        )

        batch_cls_target = rpn_cls_target[
            (rpn_cls_target == 0.) | (rpn_cls_target == 1.)
        ]

        # Check that the non negative target class are exactly the size
        # as the minibatch
        self.assertEqual(
            batch_cls_target.shape,
            (self.config.target.minibatch_size, )
        )

        # Check that we get upto foreground_fraction of positive anchors.
        self.assertLessEqual(
            batch_cls_target[batch_cls_target == 1.].shape[0] /
            batch_cls_target.shape[0],
            self.config.target.foreground_fraction
        )

    def testTypes(self):
        """Tests that return types are the expected ones.
        """
        # We repeat testBasic's setup.
        model = RPN(
            self.num_anchors, self.config, debug=True
        )
        pretrained_output_shape = (1, 32, 32, 512)
        pretrained_output = tf.placeholder(
            tf.float32, shape=pretrained_output_shape)

        image_shape_val = (
            int(pretrained_output_shape[1] * self.stride),
            int(pretrained_output_shape[2] * self.stride),
        )

        gt_boxes_shape = (4, 4)
        gt_boxes = tf.placeholder(tf.float32, shape=gt_boxes_shape)
        image_shape_shape = (2,)
        image_shape = tf.placeholder(tf.float32, shape=image_shape_shape)

        total_anchors = (
            pretrained_output_shape[1] * pretrained_output_shape[2] *
            self.num_anchors
        )
        all_anchors_shape = (total_anchors, 4)
        all_anchors = tf.placeholder(tf.float32, shape=all_anchors_shape)
        layers = model(
            pretrained_output, image_shape, all_anchors, gt_boxes=gt_boxes
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            layers_inst = sess.run(layers, feed_dict={
                pretrained_output: np.random.rand(
                    *pretrained_output_shape
                ),
                gt_boxes: generate_gt_boxes(
                    gt_boxes_shape[0], image_shape_val
                ),
                all_anchors: generate_anchors(
                    generate_anchors_reference(
                        self.base_size, self.ratios, self.scales
                    ),
                    16,
                    pretrained_output_shape[1:3]
                ),
                image_shape: image_shape_val,
            })

        # Assertions
        proposals = layers_inst['proposals']
        scores = layers_inst['scores']
        rpn_cls_prob = layers_inst['rpn_cls_prob']
        rpn_cls_score = layers_inst['rpn_cls_score']
        rpn_bbox_pred = layers_inst['rpn_bbox_pred']
        rpn_cls_target = layers_inst['rpn_cls_target']
        rpn_bbox_target = layers_inst['rpn_bbox_target']
        # Everything should have dtype=tf.float32
        self.assertAllEqual(
            # We have 7 values we want to compare to tf.float32.
            [tf.float32] * 7,
            [
                proposals.dtype, scores.dtype, rpn_cls_prob.dtype,
                rpn_cls_score.dtype, rpn_bbox_pred.dtype,
                rpn_cls_target.dtype, rpn_bbox_target.dtype,
            ]

        )

    def testLoss(self):
        """Tests that loss returns reasonable values in simple cases.
        """
        model = RPN(
            self.num_anchors, self.config, debug=True
        )

        # Define placeholders that are used inside the loss method.
        rpn_cls_prob = tf.placeholder(tf.float32)
        rpn_cls_target = tf.placeholder(tf.float32)
        rpn_cls_score = tf.placeholder(tf.float32)
        rpn_bbox_target = tf.placeholder(tf.float32)
        rpn_bbox_pred = tf.placeholder(tf.float32)

        loss = model.loss({
            'rpn_cls_prob': rpn_cls_prob,
            'rpn_cls_target': rpn_cls_target,
            'rpn_cls_score': rpn_cls_score,
            'rpn_bbox_target': rpn_bbox_target,
            'rpn_bbox_pred': rpn_bbox_pred,
        })

        # Test perfect score.
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_dict = sess.run(loss, feed_dict={
                # Probability is (background_prob, foreground_prob)
                rpn_cls_prob: [[0, 1], [1., 0]],
                # Target: 1 being foreground, 0 being background.
                rpn_cls_target: [1, 0],
                # Class scores before applying softmax. Since using cross
                # entropy, we need a big difference between values.
                rpn_cls_score: [[-100., 100.], [100., -100.]],
                # Targets and predictions are exactly equal.
                rpn_bbox_target: [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
                rpn_bbox_pred: [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
            })

            # Assert close since cross-entropy could return very small value.
            self.assertAllClose(tuple(loss_dict.values()), (0, 0))


if __name__ == "__main__":
    tf.test.main()
