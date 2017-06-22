import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .pretrained import Pretrained


class VGG(Pretrained):

    def __init__(self, trainable=True, name='vgg'):
        super(VGG, self).__init__(name=name)
        self._trainable = trainable

    def _build(self, inputs):
        """
        args:
            input: a Tensor of shape [batch_size, height, width, channels]

        output:
        """
        net = slim.repeat(
            inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1',
            trainable=self._trainable)
        net = slim.max_pool2d(net, kernel_size=[2, 2], padding='VALID', scope='pool1')
        net = slim.repeat(
            net, 2, slim.conv2d, 128, [3, 3], scope='conv2',
            trainable=self._trainable)
        net = slim.max_pool2d(net, kernel_size=[2, 2], padding='VALID', scope='pool2')
        net = slim.repeat(
            net, 3, slim.conv2d, 256, [3, 3], scope='conv3',
            trainable=self._trainable)
        net = slim.max_pool2d(net, kernel_size=[2, 2], padding='VALID', scope='pool3')
        net = slim.repeat(
            net, 3, slim.conv2d, 512, [3, 3], scope='conv4',
            trainable=self._trainable)
        net = slim.max_pool2d(net, kernel_size=[2, 2], padding='VALID', scope='pool4')
        net = slim.repeat(
            net, 3, slim.conv2d, 512, [3, 3], scope='conv5',
            trainable=self._trainable)
        return net
