import sonnet as snt
import tensorflow as tf
import numpy as np

from sonnet.testing.parameterized import parameterized

from vgg import VGG

class VGGTest(parameterized.ParameterizedTestCase, tf.test.TestCase):
    def setUp(self):
        super(VGGTest, self).setUp()

    def testBasic(self):
        model = VGG()

        batch_image_placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        intermediate_layer = model(batch_image_placeholder)

        with self.test_session() as sess:
            # As in the case of a real session we need to initialize the variables.
            sess.run(tf.global_variables_initializer())
            width = np.random.randint(500, 600)
            height = np.random.randint(500, 600)
            out = sess.run(intermediate_layer, feed_dict={
                batch_image_placeholder: np.random.rand(1, width, width, 3)
            })
            # with width and height between 500 and 600 we should have this output
            self.assertEqual(out.shape, (1, 32, 32, 512))


if __name__ == "__main__":
    tf.test.main()
