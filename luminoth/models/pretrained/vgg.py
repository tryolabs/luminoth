from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.nets import vgg

from .pretrained import Pretrained

DEFAULT_ENDPOINT = 'resnet_v2_101/block4/unit_3/bottleneck_v2/conv3'


class VGG(Pretrained):

    def __init__(self, trainable=True, endpoint=DEFAULT_ENDPOINT, name='vgg'):
        super(VGG, self).__init__(name=name)
        self._trainable = trainable
        self._endpoint = endpoint

    def _build(self, inputs, is_training=False):
        inputs = self._preprocess(inputs)
        with arg_scope(vgg.vgg_arg_scope()):
            _, end_points = vgg.vgg_16(
                inputs, is_training=self._trainable, spatial_squeeze=False
            )
            return {
                # TODO: Fix fasterrcnn_1/vgg scope problem
                'net': dict(end_points)[
                    '{}/{}'.format(self.module_name, self._endpoint)
                ],
            }

    def _preprocess(self, inputs):
        inputs = self._substract_channels(inputs)
        return inputs
