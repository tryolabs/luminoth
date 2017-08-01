from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.nets import vgg

from .pretrained import Pretrained

DEFAULT_ENDPOINT = 'vgg_16/conv5/conv5_1'


class VGG(Pretrained):

    def __init__(self, config, name='vgg'):
        super(VGG, self).__init__(name=name)
        self._trainable = config.trainable
        self._endpoint = config.endpoint or DEFAULT_ENDPOINT
        self._finetune_num_layers = config.finetune_num_layers

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
