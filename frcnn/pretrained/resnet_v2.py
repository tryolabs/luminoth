import tensorflow as tf
import sonnet as snt

from .pretrained import Pretrained
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.nets import resnet_v2

class ResNetV2(Pretrained):
    def __init__(self, trainable=True, name='resnet_v2'):
        super(ResNetV2, self).__init__(name=name)
        self._trainable = trainable

    def _build(self, inputs, is_training=False):
        is_training = self._trainable and is_training
        inputs = self._preprocess(inputs)
        with arg_scope(resnet_v2.resnet_utils.resnet_arg_scope(is_training=is_training)):
            net, end_points = resnet_v2.resnet_v2_101(inputs)

        return {
            'net': dict(end_points)['fasterrcnn_1/resnet_v2/resnet_v2_101/block4/unit_3/bottleneck_v2/conv3']
        }

    def _preprocess(self, inputs):
        inputs = self._substract_channels(inputs)
        return inputs