import sonnet as snt
import tensorflow as tf


class Pretrained(snt.AbstractModule):
    def __init__(self, name='pretrained'):
        super(Pretrained, self).__init__(name=name)

    def _load_weights(self, checkpoint_file, old_prefix=None, new_prefix=None):
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

        module_variables = snt.get_variables_in_module(self, tf.GraphKeys.GLOBAL_VARIABLES)
        assert len(module_variables) > 0

        load_variables = []
        variables = [(v, v.op.name) for v in module_variables]
        for var, var_name in variables:
            if old_prefix is not None and new_prefix is not None:
                checkpoint_var_name = var_name.replace(new_prefix, old_prefix)
            else:
                checkpoint_var_name = var_name

            var_value = tf.contrib.framework.load_variable(checkpoint_file, checkpoint_var_name)
            load_variables.append(
                tf.assign(var, var_value)
            )

        tf.logging.debug('Loading {} variables from pretrained checkpoint {}'.format(
            len(load_variables), checkpoint_file
        ))

        load_op = tf.group(*load_variables)

        return load_op