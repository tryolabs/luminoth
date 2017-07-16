import sonnet as snt
import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


class Pretrained(snt.AbstractModule):
    def __init__(self, name='pretrained'):
        super(Pretrained, self).__init__(name=name)

    def load_weights(self, checkpoint_file):
        """
        Creates operations to load weigths from checkpoint for each of the
        variables defined in the module. It is assumed that all variables
        of the module are included in the checkpoint but with a different prefix.

        TODO: We should get a better way to "translate" variables names from checkpoint to module variables.

        Args:
            checkpoint_file: Path to checkpoint file.
            old_prefix: string to replace
            new_prefix: string to replace with

        Returns:
            Load weights operation
        """
        if checkpoint_file is None:
            return tf.no_op(name='not_loading_pretrained')

        module_variables = snt.get_variables_in_module(
            self, tf.GraphKeys.GLOBAL_VARIABLES
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

        tf.logging.debug(
            'Loading {} variables from pretrained checkpoint {}'.format(
                len(load_variables), checkpoint_file
            ))

        load_op = tf.group(*load_variables)

        return load_op

    def _substract_channels(self, inputs, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
        num_channels = len(means)
        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=3, values=channels)
