from tensorflow.contrib.slim.nets import resnet_v2

from .pretrained import Pretrained


class ResNetV2(Pretrained):

    DEFAULT_ENDPOINT = 'resnet_v2_101/block4/unit_3/bottleneck_v2/conv3'

    def __init__(self, config, parent_name=None, name='resnet_v2'):
        super(ResNetV2, self).__init__(
            config, parent_name=parent_name, name=name
        )

    @property
    def arg_scope(self):
        return resnet_v2.resnet_utils.resnet_arg_scope(
            is_training=self._trainable,
            weight_decay=self._weight_decay
        )

    @property
    def network(self):
        return resnet_v2.resnet_v2_101

    def _preprocess(self, inputs):
        inputs = self._substract_channels(inputs)
        return inputs
