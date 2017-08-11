import tensorflow as tf
import numpy as np

from easydict import EasyDict
from luminoth.models.pretrained.pretrained import (
    Pretrained, _R_MEAN, _G_MEAN, _B_MEAN
)
from luminoth.models.pretrained.vgg import VGG


class PretrainedTest(tf.test.TestCase):
    def testAbstractModule(self):
        """
        Cannot create instance of Pretrained itself.
        """
        with self.assertRaises(TypeError):
            Pretrained()

    def testSubstractChannels(self):
        m = VGG(EasyDict({
            'trainable': False,
            'finetune_num_layers': 0,
            'weight_decay': 0,
            'endpoint': None,
        }))
        inputs = tf.placeholder(tf.float32, [1, 2, 2, 3])
        substracted_inputs = m._substract_channels(inputs)
        inputs
        # white image
        r = 255. - _R_MEAN
        g = 255. - _G_MEAN
        b = 255. - _B_MEAN
        with self.test_session() as sess:
            res = sess.run(substracted_inputs, feed_dict={
                inputs: np.ones([1, 2, 2, 3]) * 255
            })
            # Assert close and not equals because of floating point differences
            # between TF and numpy
            self.assertAllClose(
                res,
                # numpy broadcast multiplication
                np.ones([1, 2, 2, 3]) * [r, g, b]
            )


if __name__ == '__main__':
    tf.test.main()
