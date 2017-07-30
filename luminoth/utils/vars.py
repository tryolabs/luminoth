import tensorflow as tf
import sonnet as snt
import collections


VALID_INITIALIZERS = {
    'truncated_normal_initializer': tf.truncated_normal_initializer,
    'variance_scaling_initializer': tf.contrib.layers.variance_scaling_initializer,
    'random_normal_initializer': tf.random_normal_initializer,
}


def variable_summaries(var, name, collections):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections)
        num_params = tf.reduce_prod(tf.shape(var))
        tf.summary.scalar('num_params', num_params, collections)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, collections)
        tf.summary.scalar('max', tf.reduce_max(var), collections)
        tf.summary.scalar('min', tf.reduce_min(var), collections)
        tf.summary.histogram('histogram', var, collections)
        tf.summary.scalar('sparsity', tf.nn.zero_fraction(var), collections)


def get_initializer(initializer_config):
    if initializer_config.type not in VALID_INITIALIZERS:
        raise ValueError('Initializer "{}" is not valid.'.format(
            initializer_config.type))

    config = initializer_config.copy()
    initializer = VALID_INITIALIZERS[config.pop('type')]

    return initializer(**config)


def get_saver(modules, var_collections=(tf.GraphKeys.GLOBAL_VARIABLES,),
              ignore_scope=None, **kwargs):
    if not isinstance(modules, collections.Iterable):
        modules = [modules]

    variable_map = {}
    for module in modules:
        for collection in var_collections:
            model_variables = snt.get_normalized_variable_map(
                module, collection
            )
            total_model_variables = len(model_variables)
            if ignore_scope:
                model_variables = {
                    k: v for k, v in model_variables.items()
                    if ignore_scope not in k
                }
                new_total_model_variables = len(model_variables)
                tf.logging.info(
                    'Not loading/saving {} variables with scope "{}"'.format(
                        total_model_variables - new_total_model_variables,
                        ignore_scope))

            variable_map.update(model_variables)

    return tf.train.Saver(var_list=variable_map, **kwargs)
