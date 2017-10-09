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
            'network': {
                'num_classes': 20,
                'with_rcnn': True
            },
            'train': {
                'debug': True,
                'seed': None,
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
                'finetune_num_layers': 3,
                'weight_decay': 0.0005,
            },
            'rcnn': {
                'enabled': True,
                'layer_sizes': [4096, 4096],
                'dropout_keep_prop': 1.0,
                'activation_function': 'relu6',
                'l2_regularization_scale': 0.0005,
                'initializer': {
                    'type': 'variance_scaling_initializer',
                    'factor': 1.0,
                    'uniform': 'True',
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
               'initializer': {
                 'type': 'truncated_normal_initializer',
                 'mean': 0.0,
                 'stddev': 0.01
               },
               'activation_function': 'relu6',
               'l2_regularization_scale': 0.0005,
               'proposals': {
                 'pre_nms_top_n': 12000,
                 'post_nms_top_n': 2000,
                 'nms_threshold': 0.6,
                 'min_size': 0
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
        })
        self.image_size = (600, 800)
        self.image = np.random.randint(low=0, high=255, size=(1, 600, 800, 3))
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
                          self.config.network.num_classes).all()
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

        # Check that every rpn proposal has 4 coordinates + 1 batch index
        self.assertEqual(
            rpn_prediction['proposals'].shape[1],
            5
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
        config.anchors.base_size = 16
        config.anchors.scales = [0.5, 1, 2]
        config.anchors.ratios = [0.5, 1, 2]
        config.anchors.stride = 1  # image is 32 x 32

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

        # Check values are sequential.
        self._assert_sequential_values(anchors[:, 0], config.anchors.stride)
        self._assert_sequential_values(anchors[:, 1], config.anchors.stride)
        self._assert_sequential_values(anchors[:, 2], config.anchors.stride)
        self._assert_sequential_values(anchors[:, 3], config.anchors.stride)

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
