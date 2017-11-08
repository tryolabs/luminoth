import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from luminoth.models.base import BaseNetwork
from luminoth.utils.checkpoint_downloader import get_checkpoint_file

VALID_ARCHITECTURES = set([
    'vgg_16',
])


class FullyConvolutionalNetwork(BaseNetwork):
    def __init__(self, config, parent_name=None, name='fc_network',
                 **kwargs):
        super(FullyConvolutionalNetwork, self).__init__(config, name=name,
                                                        **kwargs)
        if config.get('architecture') not in VALID_ARCHITECTURES:
            raise ValueError('Invalid architecture "{}"'.format(
                config.get('architecture')
            ))

        self._architecture = config.get('architecture')
        self._config = config
        self.parent_name = parent_name

    def network(self, inputs, is_training=True):
        endpoints = {}
        if self.vgg_type:
            scope = self.module_name
            if self.parent_name:
                scope = self.parent_name + '/' + scope

            vgg_endpoints = {}
            with tf.variable_scope('vgg_16', [inputs], reuse=None):
                # Original VGG-16 blocks.
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3],
                                  scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                # Block 2.
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                                  scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                # Block 3.
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                                  scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                # Block 4.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                  scope='conv4')
                vgg_endpoints[scope + '/vgg_16/conv4/conv4_3'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                # Block 5.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                  scope='conv5')
                vgg_endpoints[scope + '/vgg_16/conv5/conv5_3'] = net
            with tf.variable_scope('vgg_16', reuse=None):
                net = slim.conv2d(net, 4096, [7, 7], padding='VALID',
                                  scope='fc6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

            # Add padding to divide by 2 without loss
            conv4_3 = vgg_endpoints[scope + '/vgg_16/conv4/conv4_3']
            paddings = [[0, 0], [1, 0], [1, 0], [0, 0]]
            conv4_3 = tf.pad(conv4_3, paddings, mode='CONSTANT')
            endpoints['vgg_16/conv4/conv4_3'] = conv4_3

            # Add padding to divide by 2 without loss
            conv5_3 = vgg_endpoints[scope + '/vgg_16/conv5/conv5_3']
            paddings = [[0, 0], [1, 0], [1, 0], [0, 0]]
            conv5_3 = tf.pad(conv5_3, paddings, mode='CONSTANT')
            net = slim.max_pool2d(conv5_3, [3, 3], stride=1, scope='pool5',
                                  padding='SAME')

            # Additional SSD blocks.
            # Block 6: Use atrous convolution
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6',
                              padding='SAME')
            endpoints['vgg_16/fc6'] = net
            # Block 7: 1x1 conv
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            endpoints['vgg_16/fc7'] = net
            return net, endpoints, vgg_endpoints

    def _build(self, inputs, is_training=True):
        inputs = self.preprocess(inputs)
        with slim.arg_scope(self.arg_scope):
            net, end_points, vgg_endpoints = self.network(
                inputs, is_training=is_training)

            return {
                'net': net,
                'end_points': end_points,
                'vgg_end_points': vgg_endpoints
            }

    def load_weights(self):
        """
        Creates operations to load weigths from checkpoint for each of the
        variables defined in the module. It is assumed that all variables
        of the module are included in the checkpoint but with a different
        prefix.

        Returns:
            load_op: Load weights operation or no_op.
        """
        if self.vgg_type:
            if self._config.get('weights') is None and \
               not self._config.get('download'):
                return tf.no_op(name='not_loading_network')

            if self._config.get('weights') is None:
                # Download the weights (or used cached) if is is not specified
                # in config file.
                # Weights are downloaded by default on the ~/.luminoth folder.
                self._config['weights'] = get_checkpoint_file(
                    self._architecture)

            module_variables = snt.get_variables_in_module(
                self, tf.GraphKeys.MODEL_VARIABLES
            )

            assert len(module_variables) > 0

            scope = self.module_name
            var_to_modify = [
                             self.module_name + '/conv6/weights',
                             self.module_name + '/conv6/biases',
                             self.module_name + '/conv7/weights',
                             self.module_name + '/conv7/biases'
                             ]

            load_variables = []
            variables = (
                [(v, v.op.name) for v in module_variables if v.op.name not in
                 var_to_modify]
            )

            variable_scope_len = len(self.variable_scope.name) + 1

            for var, var_name in variables:
                checkpoint_var_name = var_name[variable_scope_len:]
                var_value = tf.contrib.framework.load_variable(
                    self._config['weights'], checkpoint_var_name
                )
                load_variables.append(
                    tf.assign(var, var_value)
                )
            with tf.variable_scope(scope, reuse=True):
                module_variables = snt.get_variables_in_module(
                    self, tf.GraphKeys.MODEL_VARIABLES
                )
                variables = (
                    [(v, v.op.name) for v in module_variables]
                )
                # TODO: make this works
                # Original weigths and biases
                # fc6_weights = tf.get_variable(#scope +
                #                               'vgg_16/fc6/weights')
                # fc6_biases = tf.get_variable(#scope +
                #                              'vgg_16/fc6/biases')
                # fc7_weights = tf.get_variable(#scope +
                #                               'vgg_16/fc7/weights')
                # fc7_biases = tf.get_variable(#scope +
                #                              'vgg_16/fc7/biases')
                # # load_variables.append(fc6_weights)
                # # load_variables.append(fc6_biases)
                # # load_variables.append(fc7_weights)
                # # load_variables.append(fc7_biases)
                # # Weights and biases to surgery
                # block6_weights = tf.get_variable(scope +
                #                                  'conv6/weights')
                # block6_biases = tf.get_variable(scope +
                #                                 'conv6/biases')
                # block7_weights = tf.get_variable(scope +
                #                                  'conv7/weights')
                # block7_biases = tf.get_variable(scope +
                #                                 'conv7/biases')
                #
                # # surgery
                # load_variables.append(
                #     tf.assign(block6_weights, fc6_weights[::3, ::3, :, ::4]))
                # load_variables.append(
                #     tf.assign(block6_biases, fc6_biases[::4]))
                # load_variables.append(
                #     tf.assign(block7_weights, fc7_weights[:, :, ::4, ::4]))
                # load_variables.append(
                #     tf.assign(block7_biases, fc7_biases[::4]))

            tf.logging.info(
                'Constructing op to load {} variables from pretrained '
                'checkpoint {}'.format(
                    len(load_variables), self._config['weights']
                ))

            load_op = tf.group(*load_variables)
            return load_op

    def get_trainable_vars(self):
        """Get trainable vars for the network.

        TODO: Make this configurable.

        Returns:
            trainable_variables: A list of variables.
        """
        return snt.get_variables_in_module(self)
