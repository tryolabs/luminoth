import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers import initializers, utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops

from luminoth.models.base import BaseNetwork


VALID_ARCHITECTURES = set([
    'vgg_16',
])


class SSDFeatureExtractor(BaseNetwork):

    def __init__(self, config, parent_name=None, name='ssd_feature_extractor',
                 **kwargs):
        super(SSDFeatureExtractor, self).__init__(config, name=name, **kwargs)
        if config.get('architecture') not in VALID_ARCHITECTURES:
            raise ValueError('Invalid architecture "{}"'.format(
                config.get('architecture')
            ))

        self._architecture = config.get('architecture')
        self._config = config
        self.parent_name = parent_name
        self._dropout_keep_prob = config.dropout_keep_prob

    def _build(self, inputs, is_training=True):
        """
        Args:
            inputs: A Tensor of shape `(batch_size, height, width, channels)`.

        Returns:
            A dict of feature maps to be consumed by an SSD network
        """
        # TODO: Is there a better way to manage scoping in these cases?
        scope = self.module_name
        if self.parent_name:
            scope = self.parent_name + '/' + scope

        base_net_endpoints = super(SSDFeatureExtractor, self)._build(
            inputs, is_training=is_training)['end_points']

        # The original SSD paper uses a modified version of the vgg16 network,
        # which we'll build here
        if self.vgg_type:
            base_network_truncation_endpoint = base_net_endpoints[
                scope + '/vgg_16/conv5/conv5_3']

            # We'll add the feature maps to a collection. In the paper they use
            # one of vgg16's layers as a feature map, so we start by adding it.
            vgg_conv4_3_name = scope + '/vgg_16/conv4/conv4_3'
            vgg_conv4_3 = base_net_endpoints[vgg_conv4_3_name]

            # As it is pointed out in SSD and ParseNet papers, `conv4_3` has a
            # different features scale compared to other layers, to adjust it
            # we need to add a spatial normalization before adding the
            # predictors.
            with tf.variable_scope(vgg_conv4_3_name + '_norm'):
                inputs_shape = vgg_conv4_3.shape
                inputs_rank = inputs_shape.ndims
                dtype = vgg_conv4_3.dtype.base_dtype

                norm_dim = tf.range(inputs_rank - 1, inputs_rank)
                params_shape = inputs_shape[-1:]

                vgg_conv4_3_norm = tf.nn.l2_normalize(
                    vgg_conv4_3, norm_dim, epsilon=1e-12
                )

                # Post scaling.
                scale = variables.model_variable(
                    'gamma', shape=params_shape, dtype=dtype,
                    initializer=init_ops.ones_initializer()
                )

                vgg_conv4_3_norm = tf.multiply(vgg_conv4_3_norm, scale)

            tf.add_to_collection('FEATURE_MAPS', vgg_conv4_3_norm)

            # TODO: check that the usage of `padding='VALID'` is correct
            # TODO: check that the 1x1 convs actually use relu
            # Modifications to vgg16
            with tf.variable_scope('extra_feature_layers'):
                net = slim.max_pool2d(base_network_truncation_endpoint, [3, 3],
                                      padding='SAME', stride=1, scope='pool5')
                # TODO: or is it rate=12?
                net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
                net = slim.dropout(net, self._dropout_keep_prob, is_training=is_training)
                net = slim.conv2d(net, 1024, [1, 1], scope='conv7',
                                  outputs_collections='FEATURE_MAPS')
                net = slim.dropout(net, self._dropout_keep_prob, is_training=is_training)
                net = slim.conv2d(net, 256, [1, 1], scope='conv8_1')
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv8_2',
                                  outputs_collections='FEATURE_MAPS')
                net = slim.conv2d(net, 128, [1, 1], scope='conv9_1')
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv9_2',
                                  outputs_collections='FEATURE_MAPS')
                net = slim.conv2d(net, 128, [1, 1], scope='conv10_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv10_2',
                                  padding='VALID',
                                  outputs_collections='FEATURE_MAPS')
                net = slim.conv2d(net, 128, [1, 1], scope='conv11_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv11_2',
                                  padding='VALID',
                                  outputs_collections='FEATURE_MAPS')

            # This parameter determines onto which variables we try to load the
            # pretrained weights
            self.pretrained_weights_scope = 'ssd/ssd_feature_extractor/vgg_16'

        # Its actually an ordered dict
        return utils.convert_collection_to_dict('FEATURE_MAPS')
