from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim import arg_scope

from .pretrained import Pretrained


class VGG(Pretrained):

    def __init__(self, trainable=True, name='vgg'):
        super(VGG, self).__init__(name=name)
        self._trainable = trainable

    def _build(self, inputs, is_training=False):
        inputs = self._preprocess(inputs)
        with arg_scope(vgg.vgg_arg_scope()):
            _, end_points = vgg.vgg_16(
                inputs, is_training=self._trainable, spatial_squeeze=False
            )
            return {
                # TODO: Fix fasterrcnn_1/vgg scope problem
                'net': dict(end_points)['fasterrcnn_1/vgg/vgg_16/conv5/conv5_1'],
            }

    def _preprocess(self, inputs):
        inputs = self._substract_channels(inputs)
        return inputs
