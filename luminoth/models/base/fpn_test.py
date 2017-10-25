import tensorflow as tf
import numpy as np

from luminoth.models.base.fpn import FPN
from luminoth.models.base import BaseNetwork
from easydict import EasyDict


class FPNTest(tf.test.TestCase):
    def setUp(self):
        super(FPNTest, self).setUp()
        self._config = EasyDict({
            'architecture': 'vgg_16',
            'endpoints': [
                'conv5/conv5_3',
                'conv4/conv4_3',
                'conv3/conv3_3',
            ],
            'num_channels': 256,
            'train_base': False
        })

    def _shape_size(self, level):
        shape = level.shape
        return shape[1] * shape[2]

    def _test_output_shapes(self, training):
        # We get the BaseNetwork for comparison.
        base_model = BaseNetwork(config=self._config)

        model = FPN(config=self._config)
        image_ph = tf.placeholder(tf.float32, [None, None, None, 3])
        net = model(image_ph, is_training=training)
        base_net = base_model(image_ph, is_training=training)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            random_image = np.random.rand(1, 600, 800, 3)
            fpn_levels = sess.run(net, feed_dict={
                image_ph: random_image
            })
            # Test each level is smaller than the one after, following the
            # config.
            self.assertLess(
                self._shape_size(fpn_levels[0]),
                self._shape_size(fpn_levels[1])
            )
            self.assertLess(
                self._shape_size(fpn_levels[1]),
                self._shape_size(fpn_levels[2])
            )

            self.assertEqual(
                fpn_levels[0].shape[3], self._config.num_channels
            )
            self.assertEqual(
                fpn_levels[0].shape[3], fpn_levels[1].shape[3],
            )
            self.assertEqual(
                fpn_levels[1].shape[3], fpn_levels[2].shape[3],
            )

            # Assertions in comparison with the BaseNetwork.
            base_dict = sess.run(base_net, feed_dict={
                image_ph: random_image
            })
            base_end_points_unfiltered = base_dict['end_points']
            base_end_points = []

            for ep in self._config.endpoints:
                ep = "base_network/{}/{}".format(self._config.architecture, ep)
                base_end_points.append(base_end_points_unfiltered[ep])

            # Assert FPN levels are the same size as endpoints.
            for i in range(len(base_end_points)):
                self.assertAllEqual(
                    base_end_points[i].shape[1:3],
                    fpn_levels[i].shape[1:3]
                )

    def testOutputShapesNotTraining(self):
        self._test_output_shapes(training=False)

    def testOutputShapesTraining(self):
        self._test_output_shapes(training=True)


if __name__ == '__main__':
    tf.test.main()
