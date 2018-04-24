import sonnet as snt
import tensorflow as tf

from sonnet.python.modules.conv import Conv2D
from tensorflow.contrib.layers.python.layers import utils

from luminoth.models.base import BaseNetwork


VALID_SSD_ARCHITECTURES = set([
    'truncated_vgg_16',
    'resnet_v1_101',
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

    def _init_resnet101_extra_layers(self):
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

        if self.truncated_vgg_16_type:
            # As it is pointed out in SSD and ParseNet papers, `conv4_3` has a
            # different features scale compared to other layers, to adjust it
            # we need to add a spatial normalization before adding the
            # predictors.
            vgg_conv4_3 = base_net_endpoints[scope + '/vgg_16/conv4/conv4_3']
            tf.summary.histogram('conv4_3_hist', vgg_conv4_3)
            with tf.variable_scope('conv_4_3_norm'):
                # Normalize through channels dimension (dim=3)
                vgg_conv4_3_norm = tf.nn.l2_normalize(
                    vgg_conv4_3, 3, epsilon=1e-12
                )
                # Scale.
                scale_initializer = tf.ones(
                    [1, 1, 1, vgg_conv4_3.shape[3]]
                ) * 20.0  # They initialize to 20.0 in paper
                scale = tf.get_variable(
                    'gamma',
                    dtype=vgg_conv4_3.dtype.base_dtype,
                    initializer=scale_initializer
                )
                vgg_conv4_3_norm = tf.multiply(vgg_conv4_3_norm, scale)
                tf.summary.histogram('conv4_3_normalized_hist', vgg_conv4_3)
            tf.add_to_collection('FEATURE_MAPS', vgg_conv4_3_norm)

            # The original SSD paper uses a modified version of the vgg16
            # network, which we'll modify here
            vgg_network_truncation_endpoint = base_net_endpoints[
                scope + '/vgg_16/conv5/conv5_3']
            tf.summary.histogram(
                'conv5_3_hist',
                vgg_network_truncation_endpoint
            )

            # Extra layers for vgg16 as detailed in paper
            with tf.variable_scope('extra_feature_layers'):
                self._init_vgg16_extra_layers()
                net = tf.nn.max_pool(
                    vgg_network_truncation_endpoint, [1, 3, 3, 1],
                    padding='SAME', strides=[1, 1, 1, 1], name='pool5'
                )
                net = self.conv6(net)
                net = self.activation_fn(net)
                net = self.conv7(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv7_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)
                net = self.conv8_1(net)
                net = self.activation_fn(net)
                net = self.conv8_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv8_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)
                net = self.conv9_1(net)
                net = self.activation_fn(net)
                net = self.conv9_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv9_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)
                net = self.conv10_1(net)
                net = self.activation_fn(net)
                net = self.conv10_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv10_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)
                net = self.conv11_1(net)
                net = self.activation_fn(net)
                net = self.conv11_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv11_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)

            # This parameter determines onto which variables we try to load the
            # pretrained weights
            self.pretrained_weights_scope = scope + '/vgg_16'

        elif self.resnet_type:
            # Select which resnet layers will be used as feature maps
            feat_map_1 = base_net_endpoints[
                scope + '/resnet_v1_101/block1/unit_3/bottleneck_v1/conv1']
            tf.add_to_collection('FEATURE_MAPS', feat_map_1)
            tf.summary.histogram('block1_conv1_hist', feat_map_1)

            feat_map_2 = base_net_endpoints[
                scope + '/resnet_v1_101/block1']
            tf.add_to_collection('FEATURE_MAPS', feat_map_2)
            tf.summary.histogram('block1_hist', feat_map_2)

            feat_map_3 = base_net_endpoints[
                scope + '/resnet_v1_101/block2']
            tf.add_to_collection('FEATURE_MAPS', feat_map_3)
            tf.summary.histogram('block2_hist', feat_map_3)

            feat_map_4 = base_net_endpoints[
                scope + '/resnet_v1_101/block3']
            tf.add_to_collection('FEATURE_MAPS', feat_map_4)
            tf.summary.histogram('block3_hist', feat_map_4)

            # Select truncation resnet layer
            resnet_truncation_endpoint = base_net_endpoints[
                scope + '/resnet_v1_101/block4']
            tf.add_to_collection('FEATURE_MAPS', resnet_truncation_endpoint)
            tf.summary.histogram('block4_hist', resnet_truncation_endpoint)

            # Add new feature map layers
            with tf.variable_scope('extra_feature_layers'):
                self._init_resnet101_extra_layers()

                net = self.conv9_1(resnet_truncation_endpoint)
                net = self.activation_fn(net)
                net = self.conv9_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv9_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)

                net = self.conv10_1(net)
                net = self.activation_fn(net)
                net = self.conv10_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv10_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)

                net = self.conv11_1(net)
                net = self.activation_fn(net)
                net = self.conv11_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv11_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)

            # This parameter determines onto which variables we try to load the
            # pretrained weights
            self.pretrained_weights_scope = scope + '/resnet_v1_101'

        # NOTE: I resort to this ugly hack cause tensors with aliases are
        #       getting added twice to the feature maps collection. This seems
        #       to be a tensorflow problem.
        feat_map_dict = utils.convert_collection_to_dict('FEATURE_MAPS')
        feat_map_dict.pop('ssd/ssd_feature_extractor/resnet_v1_101/'
                          'block1/unit_3/bottleneck_v1')
        feat_map_dict.pop('ssd/ssd_feature_extractor/resnet_v1_101/'
                          'block2/unit_4/bottleneck_v1')
        feat_map_dict.pop('ssd/ssd_feature_extractor/resnet_v1_101/'
                          'block3/unit_23/bottleneck_v1')
        feat_map_dict.pop('ssd/ssd_feature_extractor/resnet_v1_101/'
                          'block4/unit_3/bottleneck_v1')

        return feat_map_dict  # It's actually an ordered dict

    def get_trainable_vars(self):
        """
        Returns a list of the variables that are trainable.

        Returns:
            trainable_variables: a tuple of `tf.Variable`.
        """
        return snt.get_variables_in_module(self)
