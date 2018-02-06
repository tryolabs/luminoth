import functools

import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.slim.nets import vgg, resnet_v2, resnet_v1

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
    """
    Convolutional Neural Network used for image classification, whose
    architecture can be any of the `VALID_ARCHITECTURES`.

    This class wraps the `tf.slim` implementations of these models, with some
    helpful additions.
    """

    def __init__(self, config, name='base_network'):
        super(BaseNetwork, self).__init__(name=name)
        if config.get('architecture') not in VALID_ARCHITECTURES:
            raise ValueError('Invalid architecture: "{}"'.format(
                config.get('architecture')
            ))

        self._architecture = config.get('architecture')
        self._config = config

    @property
    def arg_scope(self):
        arg_scope_kwargs = self._config.get('arg_scope', {})

        if self.vgg_type:
            return vgg.vgg_arg_scope(**arg_scope_kwargs)

        if self.resnet_type:
            # It's the same arg_scope for v1 or v2.
            return resnet_v2.resnet_utils.resnet_arg_scope(**arg_scope_kwargs)

        raise ValueError('Invalid architecture: "{}"'.format(
            self._config.get('architecture')
        ))

    def network(self, is_training=False):
        if self.vgg_type:
            return functools.partial(
                getattr(vgg, self._architecture),
                is_training=is_training,
                spatial_squeeze=self._config.get('spatial_squeeze', False)
            )
        elif self.resnet_v1_type:
            output_stride = self._config.get('output_stride')
            train_batch_norm = (
                is_training and self._config.get('train_batch_norm')
            )
            return functools.partial(
                getattr(resnet_v1, self._architecture),
                is_training=train_batch_norm,
                num_classes=None,
                global_pool=False,
                output_stride=output_stride
            )
        elif self.resnet_v2_type:
            output_stride = self._config.get('output_stride')
            return functools.partial(
                getattr(resnet_v2, self._architecture),
                is_training=is_training,
                num_classes=self._config.get('num_classes'),
                output_stride=output_stride,
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

    @property
    def default_image_size(self):
        # Usually 224, but depends on the architecture.
        if self.vgg_type:
            return vgg.vgg_16.default_image_size
        if self.resnet_v1_type:
            return resnet_v1.resnet_v1.default_image_size
        if self.resnet_v2_type:
            return resnet_v2.resnet_v2.default_image_size

    def _build(self, inputs, is_training=False):
        inputs = self.preprocess(inputs)
        with slim.arg_scope(self.arg_scope):
            net, end_points = self.network(is_training=is_training)(inputs)

            return {
                'net': net,
                'end_points': end_points,
            }

    def preprocess(self, inputs):
        if self.vgg_type or self.resnet_type:
            inputs = self._subtract_channels(inputs)

        return inputs

    def _subtract_channels(self, inputs, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
        """Subtract channels from images.

        It is common for CNNs to subtract the mean of all images from each
        channel. In the case of RGB images we first calculate the mean from
        each of the channels (Red, Green, Blue) and subtract those values
        for training and for inference.

        Args:
            inputs: A Tensor of images we want to normalize. Its shape is
                (1, height, width, num_channels).
            means: A Tensor of shape (num_channels,) with the means to be
                subtracted from each channels on the inputs.

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
        Creates operations to load weights from checkpoint for each of the
        variables defined in the module. It is assumed that all variables
        of the module are included in the checkpoint but with a different
        prefix.

        Returns:
            load_op: Load weights operation or no_op.
        """
        if self._config.get('weights') is None and \
           not self._config.get('download'):
            return tf.no_op(name='not_loading_base_network')

        if self._config.get('weights') is None:
            # Download the weights (or used cached) if not specified in the
            # config file.
            # Weights are downloaded by default to the ~/.luminoth folder if
            # running locally, or to the job bucket if running in Google Cloud.
            self._config['weights'] = get_checkpoint_file(self._architecture)

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
                self._config['weights'], checkpoint_var_name
            )
            load_variables.append(
                tf.assign(var, var_value)
            )

        tf.logging.info(
            'Constructing op to load {} variables from pretrained '
            'checkpoint {}'.format(
                len(load_variables), self._config['weights']
            ))

        load_op = tf.group(*load_variables)

        return load_op

    def get_trainable_vars(self):
        """
        Returns a list of the variables that are trainable.

        If a value for `fine_tune_from` is specified in the config, only the
        variables starting from the first that contains this string in its name
        will be trainable. For example, specifying `vgg_16/fc6` for a VGG16
        will set only the variables in the fully connected layers to be
        trainable.
        If `fine_tune_from` is None, then all the variables will be trainable.

        Returns:
            trainable_variables: a tuple of `tf.Variable`.
        """
        all_variables = snt.get_variables_in_module(self)

        fine_tune_from = self._config.get('fine_tune_from')
        if fine_tune_from is None:
            return all_variables

        # Get the index of the first trainable variable
        var_iter = enumerate(v.name for v in all_variables)
        try:
            index = next(i for i, name in var_iter if fine_tune_from in name)
        except StopIteration:
            raise ValueError(
                '"{}" is an invalid value of fine_tune_from for this '
                'architecture.'.format(fine_tune_from)
            )

        return all_variables[index:]
