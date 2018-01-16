import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.slim.nets import resnet_utils, resnet_v1
from luminoth.models.base import BaseNetwork


DEFAULT_ENDPOINTS = {
    'resnet_v1_50': 'block3',
    'resnet_v1_101': 'block3',
    'resnet_v1_152': 'block3',
    'resnet_v2_50': 'block3',
    'resnet_v2_101': 'block3',
    'resnet_v2_152': 'block3',
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

    def __init__(self, config, name='truncated_base_network', **kwargs):
        super(TruncatedBaseNetwork, self).__init__(config, name=name, **kwargs)
        self._endpoint = (
            config.endpoint or DEFAULT_ENDPOINTS[config.architecture]
        )
        self._scope_endpoint = '{}/{}/{}'.format(
            self.module_name, config.architecture, self._endpoint
        )
        self._freeze_tail = config.freeze_tail
        self._use_tail = config.use_tail

    def _build(self, inputs, is_training=False):
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

        return self._get_endpoint(dict(pred['end_points']))

    def _build_tail(self, inputs, is_training=False):
        if not self._use_tail:
            return inputs

        if self._architecture == 'resnet_v1_101':
            train_batch_norm = (
                is_training and self._config.get('train_batch_norm')
            )
            with self._enter_variable_scope():
                weight_decay = (
                    self._config.get('arg_scope', {}).get('weight_decay', 0)
                )
                with tf.variable_scope(self._architecture, reuse=True):
                    resnet_arg_scope = resnet_utils.resnet_arg_scope(
                            batch_norm_epsilon=1e-5,
                            batch_norm_scale=True,
                            weight_decay=weight_decay
                        )
                    with slim.arg_scope(resnet_arg_scope):
                        with slim.arg_scope(
                            [slim.batch_norm], is_training=train_batch_norm
                        ):
                            blocks = [
                                resnet_utils.Block(
                                    'block4',
                                    resnet_v1.bottleneck,
                                    [{
                                        'depth': 2048,
                                        'depth_bottleneck': 512,
                                        'stride': 1
                                    }] * 3
                                )
                            ]
                            proposal_classifier_features = (
                                resnet_utils.stack_blocks_dense(inputs, blocks)
                            )
        else:
            proposal_classifier_features = inputs

        return proposal_classifier_features

    def get_trainable_vars(self):
        """
        Returns a list of the variables that are trainable.

        Returns:
            trainable_variables: a tuple of `tf.Variable`.
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
            # Resulting `trainable_vars` is empty, possibly due to the
            # `fine_tune_from` starting after the endpoint.
            trainable_vars = tuple()
        else:
            trainable_vars = all_trainable[:index + 1]

        if self._use_tail and not self._freeze_tail:
            if self._architecture == 'resnet_v1_101':
                # Retrieve the trainable vars out of the tail.
                # TODO: Tail should be configurable too, to avoid hard-coding
                # the trainable portion to `block4` and allow using something
                # in block4 as endpoint.
                var_iter = enumerate(v.name for v in all_trainable)
                try:
                    index = next(i for i, name in var_iter if 'block4' in name)
                except StopIteration:
                    raise ValueError(
                        '"block4" not present in the trainable vars retrieved '
                        'from base network.'
                    )
                trainable_vars += all_trainable[index:]

        return trainable_vars

    def _get_endpoint(self, endpoints):
        """
        Returns the endpoint tensor from the list of possible endpoints.

        Since we already have a dictionary with variable names we should be
        able to get the desired tensor directly. Unfortunately the variable
        names change with scope and the scope changes between TensorFlow
        versions. We opted to just select the tensor for which the variable
        name ends with the endpoint name we want (it should be just one).

        Args:
            endpoints: a dictionary with {variable_name: tensor}.

        Returns:
            endpoint_value: a tensor.
        """
        for endpoint_key, endpoint_value in endpoints.items():
            if endpoint_key.endswith(self._scope_endpoint):
                return endpoint_value

        raise ValueError(
            '"{}" is an invalid value of endpoint for this '
            'architecture.'.format(self._scope_endpoint)
        )
