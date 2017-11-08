import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.ssd import SSD


class SSDNetworkTest(tf.test.TestCase):

    def setUp(self):
        super(SSDNetworkTest, self).setUp()
        # Setup
        self.config = EasyDict({
            'train': {
                'debug': True,
                'seed': None,
            },
            'model': {
                'network': {
                    'num_classes': 9,
                },
                'anchors': {
                    'anchors_per_point': [4, 6, 6, 6, 6, 4],
                    'ratios': [1, 0.5, 2, 0.3, 3],
                    'min_scale': 0.2,
                    'max_scale': 0.99
                },
                'loss': {
                    'localization_loss_weight': 1.0
                },
                'base_network': {
                    'architecture': 'vgg_16',
                    'fc_endpoints': ['vgg_16/fc7'],
                    'fc_endpoints_output': [(19, 19)],
                    'hook_endpoint': 'vgg_16/fc7',
                    'trainable': True,
                    'endpoints': ['vgg_16/conv4/conv4_3'],
                    'endpoints_output': [(38, 38)],
                    'download': False,
                    'finetune_num_layers': 3,
                    'arg_scope': {
                        'weight_decay': 0.0005,
                    }
                },
                'target': {
                    'hard_negative_ratio': 3.,
                    'foreground_threshold': 0.7,
                    'background_threshold_high': 0.4,
                    'background_threshold_low': 0.0
                },
                'proposals': {
                    'class_max_detections': 100,
                    'class_nms_threshold': 0.6,
                    'total_max_detections': 300,
                    'min_prob_threshold': 0.5,
                    'filter_outside_anchors': True,
                    'nms_threshold': 0.6
                }
            }
        })

        self.image_size = (300, 300)
        self.image = np.random.randint(low=0, high=255, size=(1, 300, 300, 3))
        self.gt_boxes = np.array([
            # [10, 10, 26, 28, 1],
            # [10, 10, 20, 22, 4],
            # [10, 11, 20, 21, 1],
            # [19, 30, 31, 33, 3],
            [150, 15, 213, 55, 3],
            # [0, 30, 31, 33, 1],
            # [0, 30, 7, 33, 1],
            # [7, 7, 70, 70, 1],
            [7, 100, 299, 299, 2],
        ])
        tf.reset_default_graph()

    def _run_network(self):
        image = tf.constant(self.image, dtype=tf.float32)
        gt_boxes = tf.constant(self.gt_boxes, dtype=tf.float32)
        model = SSD(self.config)

        results = model(image, gt_boxes, is_training=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run(results)
            return results

    def _get_losses(self, config, prediction_dict, image_size):
        image = tf.constant(self.image, dtype=tf.float32)
        gt_boxes = tf.constant(self.gt_boxes, dtype=tf.float32)
        model = SSD(config)
        model(image, gt_boxes, is_training=True)
        all_losses = model.loss(prediction_dict)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            all_losses = sess.run(all_losses)
            return all_losses

    def testBasic(self):
        """
        Test basic output of the FasterCnnNetwork
        """
        results = self._run_network()

        # Should not fail
        self._get_losses(self.config, results, self.image_size)


if __name__ == "__main__":
    tf.test.main()
