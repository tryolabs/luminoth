import sonnet as snt
import tensorflow as tf

from tensorflow.contrib.slim import arg_scope

# Default RGB means used commonly.
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


class Pretrained(snt.AbstractModule):
    """Pretrained abstract module for handling basic things all pretrained
    networks share.
    """
    def __init__(self, config, parent_name=None, name='resnet_v2'):
        super(Pretrained, self).__init__(name=name)
        self._trainable = config.trainable
        self._finetune_num_layers = config.finetune_num_layers
        self._weight_decay = config.weight_decay
        self._endpoint = config.endpoint or self.DEFAULT_ENDPOINT
        self._scope_endpoint = '{}/{}'.format(
            self.module_name, self._endpoint
        )
        if parent_name:
            self._scope_endpoint = '{}/{}'.format(
                parent_name, self._scope_endpoint
            )

    def _build(self, inputs):
        inputs = self._preprocess(inputs)

        with arg_scope(self.arg_scope):
            net, end_points = self.network(inputs)

            return {
                'net': dict(end_points)[self._scope_endpoint],
            }

    def load_weights(self, checkpoint_file):
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
        if checkpoint_file is None:
            return tf.no_op(name='not_loading_pretrained')

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
                checkpoint_file, checkpoint_var_name
            )
            load_variables.append(
                tf.assign(var, var_value)
            )

        tf.logging.info(
            'Constructing op to load {} variables from pretrained '
            'checkpoint {}'.format(
                len(load_variables), checkpoint_file
            ))

        load_op = tf.group(*load_variables)

        return load_op

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
        num_channels = len(means)
        channels = tf.split(
            axis=3, num_or_size_splits=num_channels, value=inputs
        )
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=3, values=channels)

    def get_trainable_vars(self):
        """Get trainable vars for the network.

        Not all variables are trainable, it depends on the endpoint being used.
        For example, when using a Pretrained network for object detection we
        don't want to define variables below the selected endpoint to be
        trainable.

        It is also possible to partially train part of the CNN, for that case
        we use the `_finetune_num_layers` variable to define how many layers
        from the chosen endpoint we want to train.

        Returns:
            trainable_variables: A list of variables.
        """
        all_variables = snt.get_variables_in_module(self)
        var_names = [v.name for v in all_variables]
        last_idx = [
            i for i, name in enumerate(var_names) if self._endpoint in name
        ][0]

        nolimit_finetune = (
            not hasattr(self, '_finetune_num_layers') or
            self._finetune_num_layers is None
        )
        if nolimit_finetune:
            return all_variables
        else:
            return all_variables[
                last_idx - self._finetune_num_layers * 2:last_idx
            ]
