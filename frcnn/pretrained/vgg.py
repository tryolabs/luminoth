import sonnet as snt
import tensorflow as tf

slim = tf.contrib.slim


class VGG(snt.AbstractModule):
    def __init__(self, name='vgg'):
        super(VGG, self).__init__(name=name)

    def _build(self, inputs):
        """
        args:
            input: a Tensor of shape [batch_size, height, width, channels]

        output:
        """
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, kernel_size=[2, 2], padding='VALID', scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, kernel_size=[2, 2], padding='VALID', scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, kernel_size=[2, 2], padding='VALID', scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, kernel_size=[2, 2], padding='VALID', scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        return net
