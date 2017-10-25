import tensorflow as tf
import numpy as np

from luminoth.models.base.fpn import FPN
from luminoth.models.base import BaseNetwork
from easydict import EasyDict


class FPNTest(tf.test.TestCase):
    def setUp(self):
        super(FPNTest, self).setUp()
        self._config_resnet = EasyDict({
            'architecture': 'resnet_v2_101',
            'endpoints': [
                'block4/unit_3/bottleneck_v2/conv3',
                'block3/unit_23/bottleneck_v2/conv1',
                'block2/unit_4/bottleneck_v2/conv1',
                'block1/unit_3/bottleneck_v2/conv1',
            ],
            'num_channels': 256,
            'train_base': False
        })
        self._config_vgg = EasyDict({
            'architecture': 'vgg_16',
            'endpoints': [
                'conv5/conv5_3',
                'conv4/conv4_3',
                'conv3/conv3_3',
            ],
            'num_channels': 256,
            'train_base': False
        })
        self._run_count = 0

    def _shape_size(self, level):
        shape = level.shape
        return shape[1] * shape[2]

    def _test_output_shapes(self, training, config):
        # We get the BaseNetwork for comparison.
        # We need to generate a unique name, because if we repeat a name,
        # sonnet will suffix it under the hood, and we won't be able to access
        # it. We need to know the name to later access the endpoints.
        base_name = 'base_network_{}'.format(self._run_count)
        self._run_count += 1
        base_model = BaseNetwork(config=config, name=base_name)

        model = FPN(config=config)
        image_ph = tf.placeholder(tf.float32, [None, None, None, 3])
        net = model(image_ph, is_training=training)
        base_net = base_model(image_ph, is_training=training)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            random_image = np.random.rand(1, 600, 800, 3)
            fpn_levels = sess.run(net, feed_dict={
                image_ph: random_image
            })
            self.assertEqual(
                fpn_levels[0].shape[3], config.num_channels
            )
            # Test each level is smaller than the one after, following the
            # config.
            for i in range(len(config.endpoints) - 1):
                try:
                    self.assertLess(
                        self._shape_size(fpn_levels[i]),
                        self._shape_size(fpn_levels[i + 1])
                    )
                    self.assertEqual(
                        fpn_levels[i].shape[3], fpn_levels[i + 1].shape[3],
                    )
                except AssertionError as e:
                    raise AssertionError(
                        '{}\n'
                        'Levels {} and {} of {} are the offenders.'.format(
                            e,
                            i, i + 1,
                            config.architecture
                        ))

            # Assertions in comparison with the BaseNetwork.
            base_dict = sess.run(base_net, feed_dict={
                image_ph: random_image
            })
            base_end_points_unfiltered = base_dict['end_points']
            base_end_points = []

            for ep in config.endpoints:
                ep = "{}/{}/{}".format(
                    base_name, config.architecture, ep)
                base_end_points.append(base_end_points_unfiltered[ep])

            # Assert FPN levels are the same size as endpoints.
            for i in range(len(base_end_points)):
                try:
                    self.assertAllEqual(
                        base_end_points[i].shape[1:3],
                        fpn_levels[i].shape[1:3]
                    )
                except AssertionError as e:
                    raise AssertionError(
                        '{}\n'
                        'Level {} is the offender.'.format(
                            e, i
                        ))

    def testOutputShapesVGG(self):
        self._test_output_shapes(training=True, config=self._config_vgg)
        self._test_output_shapes(training=False, config=self._config_vgg)

    def testOutputShapesTraining(self):
        self._test_output_shapes(training=True, config=self._config_resnet)
        self._test_output_shapes(training=False, config=self._config_resnet)


if __name__ == '__main__':
    tf.test.main()
