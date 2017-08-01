from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.nets import resnet_v2

from .pretrained import Pretrained

DEFAULT_ENDPOINT = 'resnet_v2_101/block4/unit_3/bottleneck_v2/conv3'


class ResNetV2(Pretrained):
    def __init__(self, config, name='resnet_v2'):
        super(ResNetV2, self).__init__(name=name)
        self._trainable = config.trainable
        self._endpoint = config.endpoint or DEFAULT_ENDPOINT
        self._finetune_num_layers = config.finetune_num_layers
        self._weight_decay = config.weight_decay

    def _build(self, inputs):
        inputs = self._preprocess(inputs)
        resnet_scope = resnet_v2.resnet_utils.resnet_arg_scope(
            is_training=self._trainable,
            weight_decay=self._weight_decay
        )
        with arg_scope(resnet_scope):
            net, end_points = resnet_v2.resnet_v2_101(inputs)

        return {
            # TODO: Fix fasterrcnn_1/resnet_v2 scope problem
            'net': dict(end_points)[
                '{}/{}'.format(
                    self.module_name, self._endpoint
                )
            ],
        }

    def _preprocess(self, inputs):
        inputs = self._substract_channels(inputs)
        return inputs
