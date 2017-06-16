import sonnet as snt
import tensorflow as tf
import numpy as np

from sonnet.testing.parameterized import parameterized

from .vgg import VGG


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

    def testLoadingVariables(self):

        # TODO: Formalize? Remove test because weights? Sample checkpoint?
        checkpoint_path = 'vgg_16.ckpt'

        model = VGG()
        batch_image_placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        intermediate_layer = model(batch_image_placeholder)

        prefix_to_replace = 'vgg_16/'
        prefix_new = 'vgg/'  # in real life should be fastercnn/vgg/

        load_variables = []
        variables = [(v, v.op.name) for v in snt.get_variables_in_module(model)]
        for var, var_name in variables:
            checkpoint_var_name = var_name.replace(prefix_new, prefix_to_replace)
            var_value = tf.contrib.framework.load_variable(checkpoint_path, checkpoint_var_name)
            load_variables.append(
                tf.assign(var, var_value)
            )

        load_op = tf.group(*load_variables)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(load_op))
            width = np.random.randint(500, 600)
            height = np.random.randint(500, 600)
            out = sess.run(intermediate_layer, feed_dict={
                batch_image_placeholder: np.random.rand(1, width, width, 3)
            })
            # with width and height between 500 and 600 we should have this output
            self.assertEqual(out.shape, (1, 32, 32, 512))

            # check value for one variable and compare it with checkpoint.
            for var, var_name in variables:
                checkpoint_var_name = var_name.replace(prefix_new, prefix_to_replace)
                var_value = tf.contrib.framework.load_variable(checkpoint_path, checkpoint_var_name)
                existing_var_value = sess.run(var)
                self.assertAllEqual(var_value, existing_var_value)


if __name__ == "__main__":
    tf.test.main()
