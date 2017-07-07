import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .pretrained import Pretrained


class VGG(Pretrained):

    def __init__(self, trainable=True, name='vgg'):
        super(VGG, self).__init__(name=name)
        self._trainable = trainable

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.
        Args:
            weight_decay: The l2 regularization coefficient.
        Returns:
            An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def _build(self, inputs):
        """
        args:
            inputs: a Tensor of shape [batch_size, height, width, channels]

        output:
        """

        end_points_collection = 'end_points'

        inputs = self._preprocess(inputs)
        with slim.arg_scope(self.vgg_arg_scope()):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
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

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return {
            'net': net,
            'inputs': inputs,
            'end_points': end_points,
        }

    def _preprocess(self, inputs):
        inputs = self._substract_channels(inputs)
        return inputs

