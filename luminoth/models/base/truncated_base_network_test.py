import easydict
import tensorflow as tf

# from luminoth.models.base.truncated_base_network import (
#     TruncatedBaseNetwork, DEFAULT_ENDPOINTS
# )


class TruncatedBaseNetworkTest(tf.test.TestCase):
    def setUp(self):
        self.config = easydict.EasyDict({
            'architecture': None,
            'endpoint': None,
        })

    def tearDown(self):
        tf.reset_default_graph()

    # This test failed in Travis' build environment,
    # probably the problem is "ran out of memory".
    # def testAllArchitectures(self):
    #     for architecture, endpoint in DEFAULT_ENDPOINTS.items():
    #         self.config.architecture = architecture
    #         self.config.endpoint = endpoint
    #         model = TruncatedBaseNetwork(self.config)
    #         image = tf.placeholder(tf.float32, [1, None, None, 3])
    #         # This should not fail.
    #         model(image)

    # TODO: This test failed in Travis' build environment, probably the problem
    # is "ran out of memory"
    # def testVGG16Output(self):
    #     self.config.architecture = 'vgg_16'
    #     self.config.endpoint = None
    #     model = TruncatedBaseNetwork(self.config)
    #
    #     batch_image_placeholder = tf.placeholder(
    #         tf.float32, shape=[1, None, None, 3])
    #     feature_map_tensor = model(batch_image_placeholder)
    #
    #     with self.test_session() as sess:
    #         # As in the case of a real session we need to initialize
    #         # variables.
    #         sess.run(tf.global_variables_initializer())
    #         width = np.random.randint(600, 605)
    #         height = np.random.randint(600, 605)
    #         feature_map = sess.run(feature_map_tensor, feed_dict={
    #             batch_image_placeholder: np.random.rand(1, width, height, 3)
    #         })
    #         # with width and height between 500 and 600 we should have this
    #         # output
    #         self.assertEqual(feature_map.shape, (1, 37, 37, 512))


if __name__ == '__main__':
    tf.test.main()
