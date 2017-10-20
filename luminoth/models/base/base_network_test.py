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


if __name__ == '__main__':
    tf.test.main()
