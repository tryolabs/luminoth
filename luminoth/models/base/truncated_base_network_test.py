import easydict
import tensorflow as tf
import gc

from luminoth.models.base.truncated_base_network import (
    TruncatedBaseNetwork, DEFAULT_ENDPOINTS
)


class TruncatedBaseNetworkTest(tf.test.TestCase):
    def setUp(self):
        self.config = easydict.EasyDict({
            'architecture': None,
            'endpoint': None,
        })
        tf.reset_default_graph()

    def testAllArchitectures(self):
        for architecture, endpoint in DEFAULT_ENDPOINTS.items():
            self.config.architecture = architecture
            self.config.endpoint = endpoint
            model = TruncatedBaseNetwork(self.config)
            image = tf.placeholder(tf.float32, [1, 320, 320, 3])
            # This should not fail.
            out = model(image)
            self.assertEqual(out.get_shape()[:3], (1, 20, 20))

            # Free up memory for travis
            tf.reset_default_graph()
            gc.collect(generation=2)

    # TODO: This test fails in Travis because of OOM error.
    # def testVGG16Output(self):
    #     self.config.architecture = 'vgg_16'
    #     self.config.endpoint = None
    #     model = TruncatedBaseNetwork(self.config)

    #     batch_image_placeholder = tf.placeholder(
    #         tf.float32, shape=[1, None, None, 3])
    #     feature_map_tensor = model(batch_image_placeholder)

    #     with self.test_session() as sess:
    #         # As in the case of a real session we need to initialize
    #         # variables.
    #         sess.run(tf.global_variables_initializer())
    #         width = 192
    #         height = 192
    #         feature_map = sess.run(feature_map_tensor, feed_dict={
    #             batch_image_placeholder: np.random.rand(1, width, height, 3)
    #         })
    #         # with width and height between 200 and 200 we should have this
    #         # output
    #         self.assertEqual(
    #             feature_map.shape, (1, width / 16, height / 16, 512)
    #         )


if __name__ == '__main__':
    tf.test.main()
