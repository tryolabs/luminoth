import tensorflow as tf

from luminoth.utils.vars import variable_summaries


OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'momentum': tf.train.MomentumOptimizer,
}

LEARNING_RATE_DECAY_METHODS = set([
    'piecewise_constant', 'exponential_decay', 'none'
])


def get_learning_rate(train_config, global_step=None):
    """
    Get learning rate from train config.

    TODO: Better config usage.

    Returns:
        learning_rate: TensorFlow variable.

    Raises:
        ValueError: When the method used is not available.
    """
    method = train_config.learning_rate_decay_method

    if not method or method == 'none':
        return train_config.initial_learning_rate

    if method not in LEARNING_RATE_DECAY_METHODS:
        raise ValueError('Invalid learning_rate method "{}"'.format(method))

    if method == 'piecewise_constant':
        learning_rate = tf.train.piecewise_constant(
            global_step, boundaries=[
                tf.cast(train_config.learning_rate_decay, tf.int64), ],
            values=[
                train_config.initial_learning_rate,
                train_config.initial_learning_rate * 0.1
            ], name='learning_rate_piecewise_constant'
        )

    elif method == 'exponential_decay':
        learning_rate = tf.train.exponential_decay(
            learning_rate=train_config.initial_learning_rate,
            global_step=global_step,
            decay_steps=train_config.learning_rate_decay, decay_rate=0.96,
            staircase=True, name='learning_rate_with_decay'
        )

    tf.summary.scalar('losses/learning_rate', learning_rate)

    return learning_rate


def get_optimizer(train_config, global_step=None):
    """
    Get optimizer from train config.

    Raises:
        ValueError: When the optimizer type or learning_rate method are not
            valid.
    """
    learning_rate = get_learning_rate(train_config, global_step)
    if train_config.optimizer_type not in OPTIMIZERS:
        raise ValueError(
            'Invalid optimizer type "{}"'.format(train_config.optimizer_type)
        )

    optimizer_cls = OPTIMIZERS[train_config.optimizer_type]
    if train_config.optimizer_type == 'momentum':
        optimizer = optimizer_cls(learning_rate, train_config.momentum)
    else:
        optimizer = optimizer_cls(learning_rate)

    return optimizer


def clip_gradients_by_norm(grads_and_vars, add_to_summary=True):
    if add_to_summary:
        for grad, var in grads_and_vars:
            if grad is not None:
                variable_summaries(grad, 'grad/{}'.format(var.name[:-2]))

    # Clip by norm. Grad can be null when not training some modules.
    with tf.name_scope('clip_gradients_by_norm'):
        grads_and_vars = [
            (
                tf.check_numerics(
                    tf.clip_by_norm(gv[0], 10.),
                    'Invalid gradient'
                ), gv[1]
            )
            if gv[0] is not None else gv
            for gv in grads_and_vars
        ]

    if add_to_summary:
        for grad, var in grads_and_vars:
            if grad is not None:
                variable_summaries(
                    grad, 'clipped_grad/{}'.format(var.name[:-2]))

    return grads_and_vars
