from luminoth.models.base import BaseNetwork


DEFAULT_ENDPOINTS = {
    'resnet_v1_50': 'block3/unit_5/bottleneck_v1/conv3',
    'resnet_v1_101': 'block3/unit_22/bottleneck_v1/conv3',
    'resnet_v1_152': 'block3/unit_35/bottleneck_v1/conv3',
    'resnet_v2_50': 'block3/unit_5/bottleneck_v2/conv3',
    'resnet_v2_101': 'block3/unit_22/bottleneck_v2/conv3',
    'resnet_v2_152': 'block3/unit_35/bottleneck_v2/conv3',
    'vgg_16': 'conv5/conv5_3',
    'vgg_19': 'conv5/conv5_4',
}


class TruncatedBaseNetwork(BaseNetwork):
    """TruncatedBaseNetwork
    """
    def __init__(self, config, parent_name=None,
                 name='truncated_base_network', **kwargs):
        super(TruncatedBaseNetwork, self).__init__(config, name=name, **kwargs)
        self._endpoint = (
            config.endpoint or DEFAULT_ENDPOINTS[config.architecture]
        )
        self._parent_name = parent_name
        self._scope_endpoint = '{}/{}/{}'.format(
            self.module_name, config.architecture, self._endpoint
        )
        if parent_name:
            self._scope_endpoint = '{}/{}'.format(
                parent_name, self._scope_endpoint
            )

    def _build(self, inputs, is_training=True):
        """

        Args:
            inputs: A Tensor of shape `(batch_size, height, width, channels)`.

        Returns:
            feature_map: A Tensor of shape
                `(batch_size, feature_map_height feature_map_width, depth)`.
        """
        pred = super(TruncatedBaseNetwork, self)._build(
            inputs, is_training=is_training
        )
        return dict(pred['end_points'])[self._scope_endpoint]
