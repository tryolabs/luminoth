import tensorflow as tf

from sonnet.python.modules.conv import Conv2D
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops

from luminoth.models.base import BaseNetwork


VALID_SSD_ARCHITECTURES = set([
    'vgg_16',
])


class SSDFeatureExtractor(BaseNetwork):

    def __init__(self, config, parent_name=None, name='ssd_feature_extractor',
                 **kwargs):
        super(SSDFeatureExtractor, self).__init__(config, name=name, **kwargs)
        if self._architecture not in VALID_SSD_ARCHITECTURES:
            raise ValueError('Invalid architecture "{}"'.format(
                self._architecture
            ))
        self.parent_name = parent_name
        self.activation_fn = tf.nn.relu

    def _init_vgg16_extra_layers(self):
        # TODO: Try Xavier initializer
        self.conv6 = Conv2D(1024, [3, 3], rate=6, name='conv6')
        self.conv7 = Conv2D(1024, [1, 1], name='conv7')
        self.conv8_1 = Conv2D(256, [1, 1], name='conv8_1')
        self.conv8_2 = Conv2D(512, [3, 3], stride=2, name='conv8_2')
        self.conv9_1 = Conv2D(128, [1, 1], name='conv9_1')
        self.conv9_2 = Conv2D(256, [3, 3], stride=2, name='conv9_2')
        self.conv10_1 = Conv2D(128, [1, 1], name='conv10_1')
        self.conv10_2 = Conv2D(256, [3, 3], padding='VALID', name='conv10_2')
        self.conv11_1 = Conv2D(128, [1, 1], name='conv11_1')
        self.conv11_2 = Conv2D(256, [3, 3], padding='VALID', name='conv11_2')

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

        if self.vgg_type:
            # The original SSD paper uses a modified version of the vgg16
            # network, which we'll modify here
            vgg_network_truncation_endpoint = base_net_endpoints[
                scope + '/vgg_16/conv5/conv5_3']

            # As it is pointed out in SSD and ParseNet papers, `conv4_3` has a
            # different features scale compared to other layers, to adjust it
            # we need to add a spatial normalization before adding the
            # predictors.
            vgg_conv4_3_name = scope + '/vgg_16/conv4/conv4_3'
            vgg_conv4_3 = base_net_endpoints[vgg_conv4_3_name]
            with tf.variable_scope(vgg_conv4_3_name + '_norm'):
                inputs_shape = vgg_conv4_3.shape
                inputs_rank = inputs_shape.ndims
                dtype = vgg_conv4_3.dtype.base_dtype
                norm_dim = tf.range(inputs_rank - 1, inputs_rank)
                params_shape = inputs_shape[-1:]

                # Normalize.
                vgg_conv4_3_norm = tf.nn.l2_normalize(
                    vgg_conv4_3, norm_dim, epsilon=1e-12
                )

                # Scale.
                # TODO use tf.get_variable and initialize
                #      to 20 as described in paper
                scale = variables.model_variable(
                    'gamma', shape=params_shape, dtype=dtype,
                    initializer=init_ops.ones_initializer()
                )
                vgg_conv4_3_norm = tf.multiply(vgg_conv4_3_norm, scale)
            tf.add_to_collection('FEATURE_MAPS', vgg_conv4_3_norm)

            # Extra layers for vgg16 as detailed in paper
            self._init_vgg16_extra_layers()
            with tf.variable_scope('extra_feature_layers'):
                # from IPython import embed; embed(display_banner=False)
                net = tf.nn.max_pool(
                    vgg_network_truncation_endpoint, [1, 3, 3, 1],
                    padding='SAME', strides=[1, 1, 1, 1], name='pool5'
                )
                net = self.conv6(net)
                net = self.activation_fn(net)
                net = self.conv7(net)
                net = self.activation_fn(net)
                tf.add_to_collection('FEATURE_MAPS', net)
                net = self.conv8_1(net)
                net = self.activation_fn(net)
                net = self.conv8_2(net)
                net = self.activation_fn(net)
                tf.add_to_collection('FEATURE_MAPS', net)
                net = self.conv9_1(net)
                net = self.activation_fn(net)
                net = self.conv9_2(net)
                net = self.activation_fn(net)
                tf.add_to_collection('FEATURE_MAPS', net)
                net = self.conv10_1(net)
                net = self.activation_fn(net)
                net = self.conv10_2(net)
                net = self.activation_fn(net)
                tf.add_to_collection('FEATURE_MAPS', net)
                net = self.conv11_1(net)
                net = self.activation_fn(net)
                net = self.conv11_2(net)
                net = self.activation_fn(net)
                tf.add_to_collection('FEATURE_MAPS', net)

            # This parameter determines onto which variables we try to load the
            # pretrained weights
            self.pretrained_weights_scope = 'ssd/ssd_feature_extractor/vgg_16'

        # It's actually an ordered dict
        return utils.convert_collection_to_dict('FEATURE_MAPS')
