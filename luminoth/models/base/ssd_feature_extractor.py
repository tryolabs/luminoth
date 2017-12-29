import tensorflow as tf
import tensorflow.contrib.slim as slim

from luminoth.models.base import BaseNetwork
from tensorflow.contrib.layers.python.layers import utils

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

        inputs = self.preprocess(inputs)

        base_net_endpoints = super(SSDFeatureExtractor, self)._build(
            inputs, is_training=is_training)['end_points']

        # The original SSD paper uses a modified version of the vgg16 network,
        # which we'll build here
        if self.vgg_type:
            # TODO: there is a problem with the scope, so I hardcoded this
            #       in the meantime, check bottom of this file [1] for more info
            base_network_truncation_endpoint = base_net_endpoints[
                'ssd_feature_extractor/vgg_16/conv5/conv5_3']

            # TODO: there is a problem with the scope, so I hardcoded this
            #       in the meantime, check bottom of this file [1] for more info
            # We'll add the feature maps to a collection. In the paper they use
            # one of vgg16's layers as a feature map, so we start by adding it.
            tf.add_to_collection('FEATURE_MAPS', base_net_endpoints[
                'ssd_feature_extractor/vgg_16/conv4/conv4_3']
            )

            # TODO: check that the usage of `padding='VALID'` is correct
            # TODO: check that the 1x1 convs actually use relu
            # Modifications to vgg16
            with tf.variable_scope('extra_feature_layers'):
                net = slim.max_pool2d(base_network_truncation_endpoint, [3, 3],
                                      padding='SAME', stride=1, scope='pool5')
                # TODO: or is it rate=12?
                net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
                net = slim.conv2d(net, 1024, [1, 1], scope='conv7',
                                  outputs_collections='FEATURE_MAPS')
                net = slim.conv2d(net, 256, [1, 1], scope='conv8_1')
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv8_2',
                                  outputs_collections='FEATURE_MAPS')
                net = slim.conv2d(net, 128, [1, 1], scope='conv9_1')
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv9_2',
                                  outputs_collections='FEATURE_MAPS')
                net = slim.conv2d(net, 128, [1, 1], scope='conv10_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv10_2',
                                  padding='VALID', outputs_collections='FEATURE_MAPS')
                net = slim.conv2d(net, 128, [1, 1], scope='conv11_1')
                # import ipdb; ipdb.set_trace()
                net = slim.conv2d(net, 256, [3, 3], scope='conv11_2',
                                  padding='VALID', outputs_collections='FEATURE_MAPS')

        # Its actually an ordered dict
        return utils.convert_collection_to_dict('FEATURE_MAPS')

# [1]:
# ipdb> for k in base_net_endpoints.keys(): print(k)
# ssd_feature_extractor/vgg_16/conv1/conv1_1
# ssd_feature_extractor/vgg_16/conv1/conv1_2
# ssd/ssd_feature_extractor/vgg_16/pool1
# ssd_feature_extractor/vgg_16/conv2/conv2_1
# ssd_feature_extractor/vgg_16/conv2/conv2_2
# ssd/ssd_feature_extractor/vgg_16/pool2
# ssd_feature_extractor/vgg_16/conv3/conv3_1
# ssd_feature_extractor/vgg_16/conv3/conv3_2
# ssd_feature_extractor/vgg_16/conv3/conv3_3
# ssd/ssd_feature_extractor/vgg_16/pool3
# ssd_feature_extractor/vgg_16/conv4/conv4_1
# ssd_feature_extractor/vgg_16/conv4/conv4_2
# ssd_feature_extractor/vgg_16/conv4/conv4_3
# ssd/ssd_feature_extractor/vgg_16/pool4
# ssd_feature_extractor/vgg_16/conv5/conv5_1
# ssd_feature_extractor/vgg_16/conv5/conv5_2
# ssd_feature_extractor/vgg_16/conv5/conv5_3
# ssd/ssd_feature_extractor/vgg_16/pool5
# ssd_feature_extractor/vgg_16/fc6
# ssd_feature_extractor/vgg_16/fc7
# ssd_feature_extractor/vgg_16/fc8
