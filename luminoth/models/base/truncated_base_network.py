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
    """
    Feature extractor for images using a regular CNN.

    By using the notion of an "endpoint", we truncate a classification CNN at
    a certain layer output, and return this partial feature map to be used as
    a good image representation for other ML tasks.
    """

    def __init__(self, config, parent_name=None, name='truncated_base_network',
                 **kwargs):
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
                `(batch_size, feature_map_height, feature_map_width, depth)`.
                The resulting dimensions depend on the CNN architecture, the
                endpoint used, and the dimensions of the input images.
        """
        pred = super(TruncatedBaseNetwork, self)._build(
            inputs, is_training=is_training
        )
        try:
            return dict(pred['end_points'])[self._scope_endpoint]
        except KeyError:
            raise ValueError(
                '"{}" is an invalid value of endpoint for this '
                'architecture.'.format(self._endpoint)
            )

    def get_trainable_vars(self):
        """
        Returns a list of the variables that are trainable.

        Returns:
            trainable_variables: a list of `tf.Variable`.
        """
        all_trainable = super(TruncatedBaseNetwork, self).get_trainable_vars()

        # Get the index of the last endpoint scope variable.
        # For example, if the endpoint for ResNet-50 is set as
        # "block4/unit_3/bottleneck_v1/conv2", then it will get 155,
        # because the variables (with their indexes) are:
        #   153 block4/unit_3/bottleneck_v1/conv2/weights:0
        #   154 block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta:0
        #   155 block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma:0
        var_iter = enumerate(v.name for v in all_trainable)
        scope_var_index_iter = (
            i for i, name in var_iter if self._endpoint in name
        )
        index = None
        for index in scope_var_index_iter:
            pass

        if index is None:
            raise ValueError(
                '"{}" is an invalid value of endpoint for this '
                'architecture.'.format(self._endpoint)
            )

        return all_trainable[:index + 1]
