import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn import FasterRCNN
from luminoth.utils.bbox_transform import clip_boxes


class FasterRCNNNetworkTest(tf.test.TestCase):

    def setUp(self):
        super(FasterRCNNNetworkTest, self).setUp()
        # Setup
        self.config = EasyDict({
            'train': {
                'debug': True,
                'seed': None,
            },
            'model': {
                'network': {
                    'num_classes': 20,
                    'with_rcnn': True
                },
                'anchors': {
                    'base_size': 256,
                    'scales': [0.5, 1, 2],
                    'ratios': [0.5, 1, 2],
                    'stride': 16
                },
                'loss': {
                    'rpn_cls_loss_weight': 1.0,
                    'rpn_reg_loss_weights': 2.0,
                    'rcnn_cls_loss_weight': 1.0,
                    'rcnn_reg_loss_weights': 2.0,
                },
                'base_network': {
                    'architecture': 'vgg_16',
                    'trainable': True,
                    'endpoint': 'conv5/conv5_1',
                    'download': False,
                    'fine_tune_from': 'conv4/conv4_2',
                    'freeze_tail': False,
                    'use_tail': True,
                    'arg_scope': {
                        'weight_decay': 0.0005,
                    }
                },
                'rcnn': {
                    'enabled': True,
                    'layer_sizes': [4096, 4096],
                    'dropout_keep_prob': 1.0,
                    'activation_function': 'relu6',
                    'l2_regularization_scale': 0.0005,
                    'l1_sigma': 3.0,
                    'use_mean': False,
                    'rcnn_initializer': {
                        'type': 'variance_scaling_initializer',
                        'factor': 1.0,
                        'uniform': True,
                        'mode': 'FAN_AVG'
                    },
                    'bbox_initializer': {
                        'type': 'variance_scaling_initializer',
                        'factor': 1.0,
                        'uniform': True,
                        'mode': 'FAN_AVG'
                    },
                    'cls_initializer': {
                        'type': 'variance_scaling_initializer',
                        'factor': 1.0,
                        'uniform': True,
                        'mode': 'FAN_AVG'
                    },
                    'roi': {
                        'pooling_mode': 'crop',
                        'pooled_width': 7,
                        'pooled_height': 7,
                        'padding': 'VALID'
                    },
                    'proposals': {
                        'class_max_detections': 100,
                        'class_nms_threshold': 0.6,
                        'total_max_detections': 300,
                        'min_prob_threshold': 0.0
                    },
                    'target': {
                        'foreground_fraction': 0.25,
                        'minibatch_size': 64,
                        'foreground_threshold': 0.5,
                        'background_threshold_high': 0.5,
                        'background_threshold_low': 0.1,
                    }
                },
                'rpn': {
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
                    'activation_function': 'relu6',
                    'l2_regularization_scale': 0.0005,
                    'l1_sigma': 3.0,
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
                }
            }
        })

        self.image_size = (600, 800)
        self.image = np.random.randint(low=0, high=255, size=(600, 800, 3))
        self.gt_boxes = np.array([
            [10, 10, 26, 28, 1],
            [10, 10, 20, 22, 1],
            [10, 11, 20, 21, 1],
            [19, 30, 31, 33, 1],
        ])
        tf.reset_default_graph()

    def _run_network(self):
        image = tf.placeholder(
            tf.float32, shape=self.image.shape)
        gt_boxes = tf.placeholder(
            tf.float32, shape=self.gt_boxes.shape)
        model = FasterRCNN(self.config)

        results = model(image, gt_boxes)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run(results, feed_dict={
                gt_boxes: self.gt_boxes,
                image: self.image,
            })
            return results

    def _gen_anchors(self, config, feature_map_shape):
        model = FasterRCNN(config)
        results = model._generate_anchors(feature_map_shape)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run(results)
            return results

    def _get_losses(self, config, prediction_dict, image_size):
        image = tf.placeholder(
            tf.float32, shape=self.image.shape)
        gt_boxes = tf.placeholder(
            tf.float32, shape=self.gt_boxes.shape)
        model = FasterRCNN(config)
        model(image, gt_boxes, is_training=True)
        all_losses = model.loss(prediction_dict, return_all=True)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            all_losses = sess.run(all_losses)
            return all_losses

    def testBasic(self):
        """
        Test basic output of the FasterCnnNetwork
        """
        results = self._run_network()
        class_prediction = results['classification_prediction']
        rpn_prediction = results['rpn_prediction']

        # Check that every object is defined by 4 coordinates
        self.assertEqual(
            class_prediction['objects'].shape[1],
            4
        )

        # Check we get objects clipped to the image.
        self.assertAllEqual(
            clip_boxes(class_prediction['objects'], self.image_size),
            class_prediction['objects']
        )

        self.assertEqual(
            class_prediction['labels'].shape[0],
            class_prediction['objects'].shape[0]
        )

        # Check that every object label is less or equal than 'num_classes'
        self.assertTrue(
            np.less_equal(class_prediction['labels'],
                          self.config.model.network.num_classes).all()
        )

        # Check that the sum of class probabilities is 1
        self.assertAllClose(
            np.sum(class_prediction['rcnn']['cls_prob'], axis=1),
            np.ones((class_prediction['rcnn']['cls_prob'].shape[0]))
        )

        # Check that the sum of rpn class probabilities is 1
        self.assertAllClose(
            np.sum(rpn_prediction['rpn_cls_prob'], axis=1),
            np.ones((rpn_prediction['rpn_cls_prob'].shape[0]))
        )

        # Check that every rpn proposal has 4 coordinates
        self.assertEqual(
            rpn_prediction['proposals'].shape[1],
            4
        )

        # Check we get rpn proposals clipped to the image.
        self.assertAllEqual(
            clip_boxes(rpn_prediction['proposals'], self.image_size),
            rpn_prediction['proposals']
        )

    def testAnchors(self):
        """
        Tests about the anchors generated by the FasterRCNN
        """
        results = self._run_network()

        # Check we get anchors clipped to the image.
        self.assertAllEqual(
            clip_boxes(results['all_anchors'], self.image_size),
            results['all_anchors']
        )

        feature_map = np.random.randint(low=0, high=255, size=(1, 32, 32, 1))
        config = self.config
        config.model.anchors.base_size = 16
        config.model.anchors.scales = [0.5, 1, 2]
        config.model.anchors.ratios = [0.5, 1, 2]
        config.model.anchors.stride = 1  # image is 32 x 32

        anchors = self._gen_anchors(config, feature_map.shape)

        # Check the amount of anchors generated is correct:
        # 9216 = 32^2 * config.anchor.scales * config.anchor.ratios = 1024 * 9
        self.assertEqual(anchors.shape, (9216, 4))

        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]

        # Since we are using equal scales and ratios, the set of unique heights
        # and widths must be the same.
        self.assertAllEqual(
            np.unique(anchor_widths), np.unique(anchor_heights)
        )

        anchor_areas = anchor_widths * anchor_heights

        # We have 9 possible anchors areas, minus 3 repeated ones. 6 unique.
        self.assertAllEqual(np.unique(anchor_areas).shape[0], 6)

        # Check the anchors cover all the image.
        # TODO: Check with values calculated from config.
        self.assertEqual(np.min(anchors[:, 0]), -22)
        self.assertEqual(np.max(anchors[:, 0]), 29)

        self.assertEqual(np.min(anchors[:, 1]), -22)
        self.assertEqual(np.max(anchors[:, 1]), 29)

        self.assertEqual(np.min(anchors[:, 2]), 2)
        self.assertEqual(np.max(anchors[:, 2]), 53)

        self.assertEqual(np.min(anchors[:, 3]), 2)
        self.assertEqual(np.max(anchors[:, 3]), 53)

        stride = config.model.anchors.stride
        # Check values are sequential.
        self._assert_sequential_values(anchors[:, 0], stride)
        self._assert_sequential_values(anchors[:, 1], stride)
        self._assert_sequential_values(anchors[:, 2], stride)
        self._assert_sequential_values(anchors[:, 3], stride)

    def testLoss(self):
        """
        Tests the loss of the FasterRCNN
        """

        # Create prediction_dict's structure
        prediction_dict_random = {
            'rpn_prediction': {},
            'classification_prediction': {
                'rcnn': {
                    'cls_score': None,
                    'bbox_offsets': None
                },
                'target': {},
                '_debug': {
                    'losses': {}
                }
            }
        }
        prediction_dict_perf = {
            'rpn_prediction': {},
            'classification_prediction': {
                'rcnn': {
                    'cls_score': None,
                    'bbox_offsets': None
                },
                'target': {},
                '_debug': {
                    'losses': {}
                }
            }
        }

        # Set seeds for stable results
        rand_seed = 13
        target_seed = 43
        image_size = (60, 80)
        num_anchors = 1000

        config = EasyDict(self.config)
        config.model.rpn.l2_regularization_scale = 0.0
        config.model.rcnn.l2_regularization_scale = 0.0
        config.model.base_network.arg_scope.weight_decay = 0.0

        #   RPN

        # Random generation of cls_targets for rpn
        # where:
        #       {-1}:   Ignore
        #       { 0}:   Background
        #       { 1}:   Object
        rpn_cls_target = tf.floor(tf.random_uniform(
            [num_anchors],
            minval=-1,
            maxval=2,
            dtype=tf.float32,
            seed=target_seed,
            name=None
        ))

        # Creation of cls_scores with:
        #   score 100 in correct class
        #   score 0 in wrong class

        # Generation of opposite cls_score for rpn
        rpn_cls_score = tf.cast(
            tf.one_hot(
                tf.cast(
                    tf.mod(tf.identity(rpn_cls_target) + 1, 2),
                    tf.int32),
                depth=2,
                on_value=10),
            tf.float32
        )
        # Generation of correct cls_score for rpn
        rpn_cls_perf_score = tf.cast(
            tf.one_hot(
                tf.cast(
                    tf.identity(rpn_cls_target),
                    tf.int32),
                depth=2,
                on_value=100),
            tf.float32
        )

        # Random generation of target bbox deltas
        rpn_bbox_target = tf.floor(tf.random_uniform(
            [num_anchors, 4],
            minval=-1,
            maxval=1,
            dtype=tf.float32,
            seed=target_seed,
            name=None
        ))

        # Random generation of predicted bbox deltas
        rpn_bbox_predictions = tf.floor(tf.random_uniform(
            [num_anchors, 4],
            minval=-1,
            maxval=1,
            dtype=tf.float32,
            seed=rand_seed,
            name=None
        ))

        prediction_dict_random['rpn_prediction'][
            'rpn_cls_score'] = rpn_cls_score
        prediction_dict_random['rpn_prediction'][
            'rpn_cls_target'] = rpn_cls_target
        prediction_dict_random['rpn_prediction'][
            'rpn_bbox_target'] = rpn_bbox_target
        prediction_dict_random['rpn_prediction'][
            'rpn_bbox_pred'] = rpn_bbox_predictions

        prediction_dict_perf['rpn_prediction'][
            'rpn_cls_score'] = rpn_cls_perf_score
        prediction_dict_perf['rpn_prediction'][
            'rpn_cls_target'] = rpn_cls_target
        prediction_dict_perf['rpn_prediction'][
            'rpn_bbox_target'] = rpn_bbox_target
        prediction_dict_perf['rpn_prediction'][
            'rpn_bbox_pred'] = rpn_bbox_target

        #   RCNN

        # Set the number of classes
        num_classes = config.model.network.num_classes

        # Randomly generate the bbox_offsets for the correct class = 1
        prediction_dict_random['classification_prediction']['target'] = {
            'bbox_offsets': tf.random_uniform(
                [1, 4],
                minval=-1,
                maxval=1,
                dtype=tf.float32,
                seed=target_seed,
                name=None
            ),
            'cls': [1]
        }

        # Set the same bbox_offsets and cls for the perfect prediction
        prediction_dict_perf[
            'classification_prediction']['target'] = prediction_dict_random[
                'classification_prediction']['target'].copy()

        # Generate random scores for the num_classes + the background class
        rcnn_cls_score = tf.random_uniform(
            [1, num_classes + 1],
            minval=-100,
            maxval=100,
            dtype=tf.float32,
            seed=rand_seed,
            name=None
        )

        # Generate a perfect prediction with the correct class score = 100
        # and the rest set to 0
        rcnn_cls_perf_score = tf.cast(
            tf.one_hot(
                [1], depth=num_classes + 1,
                on_value=100
            ),
            tf.float32
        )

        # Generate the random delta prediction for each class
        rcnn_bbox_offsets = tf.random_uniform(
            [1, num_classes * 4],
            minval=-1,
            maxval=1,
            dtype=tf.float32,
            seed=rand_seed,
            name=None
        )

        # Copy the random prediction and set the correct class prediction
        # as the target one
        target_bbox_offsets = prediction_dict_random[
            'classification_prediction']['target']['bbox_offsets']
        initial_val = 1 * 4  # cls value * 4
        rcnn_bbox_perf_offsets = tf.Variable(tf.reshape(
            tf.random_uniform(
                [1, num_classes * 4],
                minval=-1,
                maxval=1,
                dtype=tf.float32,
                seed=target_seed,
                name=None
            ), [-1]))
        rcnn_bbox_perf_offsets = tf.reshape(
            tf.scatter_update(
                rcnn_bbox_perf_offsets,
                tf.range(initial_val, initial_val + 4),
                tf.reshape(target_bbox_offsets, [-1])
            ),
            [1, -1])

        prediction_dict_random['classification_prediction'][
            'rcnn']['cls_score'] = rcnn_cls_score
        prediction_dict_random['classification_prediction'][
            'rcnn']['bbox_offsets'] = rcnn_bbox_offsets

        prediction_dict_perf['classification_prediction'][
            'rcnn']['cls_score'] = rcnn_cls_perf_score
        prediction_dict_perf['classification_prediction'][
            'rcnn']['bbox_offsets'] = rcnn_bbox_perf_offsets

        loss_perfect = self._get_losses(
            config, prediction_dict_perf, image_size)
        loss_random = self._get_losses(
            config, prediction_dict_random, image_size)

        loss_random_compare = {
            'rcnn_cls_loss': 5,
            'rcnn_reg_loss': 3,
            'rpn_cls_loss': 5,
            'rpn_reg_loss': 3,
            'no_reg_loss': 16,
            'regularization_loss': 0,
            'total_loss': 22,
        }
        for loss in loss_random:
            self.assertGreaterEqual(
                loss_random[loss],
                loss_random_compare[loss],
                loss
            )
            self.assertEqual(
                loss_perfect[loss],
                0, loss
            )

    def _assert_sequential_values(self, values, delta=1):
        unique_values = np.unique(values)
        paired_values = np.column_stack(
            (unique_values[:-1], unique_values[1:])
        )
        self.assertAllEqual(
            paired_values[:, 1] - paired_values[:, 0],
            np.ones((paired_values.shape[0], ), np.int)
        )


if __name__ == "__main__":
    tf.test.main()
