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
            'freeze_tail': False,
            'use_tail': True,
            'output_stride': 16,
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

    def testTrainableVariables(self):
        inputs = tf.placeholder(tf.float32, [1, 224, 224, 3])

        model = TruncatedBaseNetwork(
            easydict.EasyDict({
                'architecture': 'resnet_v1_50',
                'endpoint': 'block4/unit_3/bottleneck_v1/conv2',
                'freeze_tail': False,
                'use_tail': True,
            })
        )
        model(inputs)
        # Variables in ResNet-50:
        # (the order of beta and gamma depends on the TensorFlow's version)
        #   0 conv1/weights:0
        #   1 conv1/BatchNorm/(beta|gamma):0
        #   2 conv1/BatchNorm/(beta|gamma):0
        #   3 block1/unit_1/bottleneck_v1/shortcut/weights:0
        #   (...)
        #   153 block4/unit_3/bottleneck_v1/conv2/weights:0
        #   154 block4/unit_3/bottleneck_v1/conv2/BatchNorm/(beta|gamma):0
        #   155 block4/unit_3/bottleneck_v1/conv2/BatchNorm/(beta|gamma):0
        #   --- endpoint ---
        #   156 block4/unit_3/bottleneck_v1/conv3/weights:0
        #   157 block4/unit_3/bottleneck_v1/conv3/BatchNorm/(beta|gamma):0
        #   158 block4/unit_3/bottleneck_v1/conv3/BatchNorm/(beta|gamma):0
        #   159 logits/weights:0
        #   160 logits/biases:0
        trainable_vars = model.get_trainable_vars()
        self.assertEqual(len(trainable_vars), 156)
        self.assertEqual(
            trainable_vars[-3].name,
            'truncated_base_network/resnet_v1_50/' +
            'block4/unit_3/bottleneck_v1/conv2/weights:0'
        )

        model = TruncatedBaseNetwork(
            easydict.EasyDict({
                'architecture': 'resnet_v1_50',
                'endpoint': 'block4/unit_2/bottleneck_v1/conv2',
                'fine_tune_from': 'block4/unit_2/bottleneck_v1/conv1',
                'freeze_tail': False,
                'use_tail': True,
            })
        )
        model(inputs)
        trainable_vars = model.get_trainable_vars()
        # Now there should be only 6 trainable vars:
        #   141 block4/unit_2/bottleneck_v1/conv1/weights:0
        #   142 block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta:0
        #   143 block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma:0
        #   144 block4/unit_2/bottleneck_v1/conv2/weights:0
        #   145 block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta:0
        #   146 block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma:0
        self.assertEqual(len(trainable_vars), 6)

        #
        # Check that we return no vars if fine_tune_from is after the chosen
        # endpoint (there is nothing to fine-tune!) and tail is frozen.
        #
        model = TruncatedBaseNetwork(
            easydict.EasyDict(
                {
                    'architecture': 'resnet_v1_50',
                    'endpoint': 'block4/unit_2/bottleneck_v1/conv1',
                    'fine_tune_from': 'block4/unit_2/bottleneck_v1/conv2',
                    'freeze_tail': True,
                    'use_tail': True,
                }
            )
        )
        model(inputs)
        self.assertEqual(len(model.get_trainable_vars()), 0)


if __name__ == '__main__':
    tf.test.main()
