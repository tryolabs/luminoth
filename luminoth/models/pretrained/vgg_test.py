import tensorflow as tf
import numpy as np

from easydict import EasyDict
from luminoth.models.pretrained import VGG


class VGGTest(tf.test.TestCase):
    def setUp(self):
        super(VGGTest, self).setUp()

    def testBasic(self):
        model = VGG(EasyDict({
            'trainable': False,
            'finetune_num_layers': 0,
            'weight_decay': 0.0,
            'endpoint': None,
        }))

        batch_image_placeholder = tf.placeholder(
            tf.float32, shape=[1, None, None, 3])
        intermediate_layer = model(batch_image_placeholder)

        with self.test_session() as sess:
            # As in the case of a real session we need to initialize variables.
            sess.run(tf.global_variables_initializer())
            width = np.random.randint(600, 605)
            height = np.random.randint(600, 605)
            out = sess.run(intermediate_layer, feed_dict={
                batch_image_placeholder: np.random.rand(1, width, height, 3)
            })
            # with width and height between 500 and 600 we should have this
            # output
            self.assertEqual(out['net'].shape, (1, 37, 37, 512))


if __name__ == '__main__':
    tf.test.main()
