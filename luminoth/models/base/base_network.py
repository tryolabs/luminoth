import sonnet as snt
import tensorflow as tf
import functools

from tensorflow.contrib.slim.nets import (
    vgg, resnet_v2, resnet_v1
)
import tensorflow.contrib.slim as slim

from luminoth.utils.checkpoint_downloader import get_checkpoint_file

# Default RGB means used commonly.
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

VALID_ARCHITECTURES = set([
    'resnet_v1_50',
    'resnet_v1_101',
    'resnet_v1_152',
    'resnet_v2_50',
    'resnet_v2_101',
    'resnet_v2_152',
    'vgg_16',
    'vgg_19',
])


class BaseNetwork(snt.AbstractModule):
    def __init__(self, config, name='base_network'):
        super(BaseNetwork, self).__init__(name=name)
        if config.architecture not in VALID_ARCHITECTURES:
            raise ValueError('Invalid architecture "{}"'.format(
                config.architecture
            ))

        self._architecture = config.architecture
        self._config = config

    @property
    def arg_scope(self):
        kwargs = {}
        weight_decay = self._config.get('weight_decay')
        if weight_decay is not None:
            kwargs['weight_decay'] = weight_decay

        if self.vgg_type:
            return vgg.vgg_arg_scope(**kwargs)
        elif self.resnet_type:
            # It's the same argscope for v1 or v2.
            return resnet_v2.resnet_utils.resnet_arg_scope(**kwargs)

    def network(self, is_training=True):
        if self.vgg_type:
            return functools.partial(
                getattr(vgg, self._architecture), spatial_squeeze=False,
                is_training=is_training
            )
        elif self.resnet_v1_type:
            return functools.partial(
                getattr(resnet_v1, self._architecture),
                is_training=is_training
            )
        elif self.resnet_v2_type:
            return functools.partial(
                getattr(resnet_v2, self._architecture),
                is_training=is_training
            )

    @property
    def vgg_type(self):
        return self._architecture.startswith('vgg')

    @property
    def resnet_type(self):
        return self._architecture.startswith('resnet')

    @property
    def resnet_v1_type(self):
        return self._architecture.startswith('resnet_v1')

    @property
    def resnet_v2_type(self):
        return self._architecture.startswith('resnet_v2')

    def _build(self, inputs, is_training=True):
        inputs = self.preprocess(inputs)
        with slim.arg_scope(self.arg_scope):
            net, end_points = self.network(is_training=is_training)(inputs)

            return {
                'net': net,
                'end_points': end_points,
            }

    def preprocess(self, inputs):
        if self.vgg_type:
            inputs = self._substract_channels(inputs)
        elif self.resnet_type:
            inputs = self._normalize(inputs)

        return inputs

    def _substract_channels(self, inputs, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
        """Substract channels from images.

        It is common for CNNs to substract the mean of all images from each
        channel. In the case of RGB images we first calculate the mean from
        each of the channels (Red, Green, Blue) and substract those values
        for training and for inference.

        Args:
            inputs: A Tensor of images we want to normalize. Its shape is
                (1, height, width, num_channels).
            means: A Tensor of shape (num_channels,) with the means to be
                substracted from each channels on the inputs.

        Returns:
            outputs: A Tensor of images normalized with the means.
                Its shape is the same as the input.
        """
        return inputs - [means]

    def _normalize(self, inputs):
        """Normalize between -1.0 to 1.0.

        Args:
            inputs: A Tensor of images we want to normalize. Its shape is
                (1, height, width, num_channels).
        Returns:
            outputs: A Tensor of images normalized between -1 and 1.
                Its shape is the same as the input.
        """
        inputs = inputs / 255.
        inputs = (inputs - 0.5) * 2.
        return inputs

    def load_weights(self):
        """
        Creates operations to load weigths from checkpoint for each of the
        variables defined in the module. It is assumed that all variables
        of the module are included in the checkpoint but with a different
        prefix.

        Args:
            checkpoint_file: Path to checkpoint file.

        Returns:
            load_op: Load weights operation or no_op.
        """
        if self._config.weights is None and not self._config.download:
            return tf.no_op(name='not_loading_base_network')

        self._config.weights = get_checkpoint_file(self._architecture)

        module_variables = snt.get_variables_in_module(
            self, tf.GraphKeys.MODEL_VARIABLES
        )
        assert len(module_variables) > 0

        load_variables = []
        variables = [(v, v.op.name) for v in module_variables]
        variable_scope_len = len(self.variable_scope.name) + 1
        for var, var_name in variables:
            checkpoint_var_name = var_name[variable_scope_len:]
            var_value = tf.contrib.framework.load_variable(
                self._config.weights, checkpoint_var_name
            )
            load_variables.append(
                tf.assign(var, var_value)
            )

        tf.logging.info(
            'Constructing op to load {} variables from pretrained '
            'checkpoint {}'.format(
                len(load_variables), self._config.weights
            ))

        load_op = tf.group(*load_variables)

        return load_op

    def get_trainable_vars(self):
        """Get trainable vars for the network.

        Not all variables are trainable, it depends on the endpoint being used.
        For example, when using a Pretrained network for object detection we
        don't want to define variables below the selected endpoint to be
        trainable.

        It is also possible to partially train part of the CNN, for that case
        we use config's `finetune_num_layers` variable to define how many
        layers from the chosen endpoint we want to train.

        Returns:
            trainable_variables: A list of variables.
        """
        all_variables = snt.get_variables_in_module(self)
        var_names = [v.name for v in all_variables]
        last_idx = [
            i for i, name in enumerate(var_names) if self._endpoint in name
        ][0]

        finetune_num_layers = self._config.get('finetune_num_layers')
        if not finetune_num_layers:
            return all_variables
        else:
            return all_variables[
                last_idx - finetune_num_layers * 2:last_idx
            ]
