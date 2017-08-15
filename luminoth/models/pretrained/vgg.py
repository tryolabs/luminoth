import functools

from tensorflow.contrib.slim.nets import vgg

from .pretrained import Pretrained


class VGG(Pretrained):

    DEFAULT_ENDPOINT = 'vgg_16/conv5/conv5_1'

    def __init__(self, config, parent_name=None, name='vgg'):
        super(VGG, self).__init__(
            config, parent_name=parent_name, name=name
        )

    @property
    def arg_scope(self):
        return vgg.vgg_arg_scope(weight_decay=self._weight_decay)

    @property
    def network(self):
        return functools.partial(
            vgg.vgg_16, is_training=self._trainable, spatial_squeeze=False
        )

    def _preprocess(self, inputs):
        inputs = self._substract_channels(inputs)
        return inputs
