import gc
import easydict
import numpy as np
import tensorflow as tf

from luminoth.models.base.base_network import (
    BaseNetwork, _R_MEAN, _G_MEAN, _B_MEAN, VALID_ARCHITECTURES
)


class BaseNetworkTest(tf.test.TestCase):

    def setUp(self):
        self.config = easydict.EasyDict({
            'architecture': 'vgg_16',
        })
        tf.reset_default_graph()

    def testDefaultImageSize(self):
        m = BaseNetwork(easydict.EasyDict({'architecture': 'vgg_16'}))
        self.assertEqual(m.default_image_size, 224)

        m = BaseNetwork(easydict.EasyDict({'architecture': 'resnet_v1_50'}))
        self.assertEqual(m.default_image_size, 224)

    def testSubtractChannels(self):
        m = BaseNetwork(self.config)
        inputs = tf.placeholder(tf.float32, [1, 2, 2, 3])
        subtracted_inputs = m._subtract_channels(inputs)
        # White image
        r = 255. - _R_MEAN
        g = 255. - _G_MEAN
        b = 255. - _B_MEAN
        with self.test_session() as sess:
            res = sess.run(subtracted_inputs, feed_dict={
                inputs: np.ones([1, 2, 2, 3]) * 255
            })
            # Assert close and not equals because of floating point
            # differences between TF and numpy
            self.assertAllClose(
                res,
                # numpy broadcast multiplication
                np.ones([1, 2, 2, 3]) * [r, g, b]
            )

    def testAllArchitectures(self):
        for architecture in VALID_ARCHITECTURES:
            self.config.architecture = architecture
            m = BaseNetwork(self.config)
            inputs = tf.placeholder(tf.float32, [1, None, None, 3])
            # Should not fail.
            m(inputs)
            # Free up memory for Travis
            tf.reset_default_graph()
            gc.collect(generation=2)

    def testTrainableVariables(self):
        inputs = tf.placeholder(tf.float32, [1, 224, 224, 3])

        model = BaseNetwork(easydict.EasyDict({'architecture': 'vgg_16'}))
        model(inputs)
        # Variables in VGG16:
        #   0 conv1/conv1_1/weights:0
        #   1 conv1/conv1_1/biases:0
        #   (...)
        #   30 fc8/weights:0
        #   31 fc8/biases:0

        self.assertEqual(len(model.get_trainable_vars()), 32)

        model = BaseNetwork(
            easydict.EasyDict(
                {'architecture': 'vgg_16', 'fine_tune_from': 'conv5/conv5_3'}
            )
        )
        model(inputs)
        # Variables from `conv5/conv5_3` to the end:
        #   conv5/conv5_3/weights:0
        #   conv5/conv5_3/biases:0
        #   fc6/weights:0
        #   fc6/biases:0
        #   fc7/weights:0
        #   fc7/biases:0
        #   fc8/weights:0
        #   fc8/biases:0
        self.assertEqual(len(model.get_trainable_vars()), 8)

        #
        # Check invalid fine_tune_from raises proper exception
        #
        model = BaseNetwork(
            easydict.EasyDict(
                {'architecture': 'vgg_16', 'fine_tune_from': 'conv5/conv99'}
            )
        )
        model(inputs)
        with self.assertRaises(ValueError):
            model.get_trainable_vars()


if __name__ == '__main__':
    tf.test.main()
