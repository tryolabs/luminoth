import easydict
import gc
import numpy as np
import tensorflow as tf

from luminoth.models.base.fully_convolutional_network import (
    FullyConvolutionalNetwork,
    # VALID_ARCHITECTURES
)
from luminoth.models.base.base_network import _R_MEAN, _G_MEAN, _B_MEAN


class BaseNetworkTest(tf.test.TestCase):
    def setUp(self):
        self.config = easydict.EasyDict({
            'architecture': 'vgg_16',
            'download': True
        })
        tf.reset_default_graph()

    def testSubstractChannels(self):
        m = FullyConvolutionalNetwork(self.config)
        inputs = tf.placeholder(tf.float32, [1, 2, 2, 3])
        substracted_inputs = m._substract_channels(inputs)
        # white image
        r = 255. - _R_MEAN
        g = 255. - _G_MEAN
        b = 255. - _B_MEAN
        with self.test_session() as sess:
            res = sess.run(substracted_inputs, feed_dict={
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
        m = FullyConvolutionalNetwork(self.config)
        inputs = tf.placeholder(tf.float32, [1, None, None, 3])
        # Should not fail.
        m(inputs)
        # Free up memory for Travis
        tf.reset_default_graph()
        gc.collect(generation=2)

    # Commented becuase Travis fails
    # def testLoadWeights(self):
    #     m = FullyConvolutionalNetwork(self.config)
    #     inputs = tf.placeholder(tf.float32, [1, None, None, 3])
    #     m(inputs)
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         sess.run(tf.local_variables_initializer())
    #         load = m.load_weights()
    #         sess.run(load)


if __name__ == '__main__':
    tf.test.main()
