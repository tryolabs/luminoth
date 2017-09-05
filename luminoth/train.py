import os

import click
import json
import tensorflow as tf

from tensorflow.python import debug as tf_debug

from luminoth.datasets import TFRecordDataset
from luminoth.models import get_model
from luminoth.utils.config import (
    load_config, merge_into, kwargs_to_config, parse_override
)
from luminoth.utils.vars import variable_summaries


OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'momentum': tf.train.MomentumOptimizer,
}

LEARNING_RATE_DECAY_METHODS = set([
    'piecewise_constant', 'exponential_decay', 'none'
])


def get_model_config(model_class, config_file, override_params, **kwargs):
    config = model_class.base_config

    # Load train extra options
    config.train = merge_into(kwargs_to_config(kwargs), config.train)

    if config_file:
        # If we have a custom config file overwriting default settings
        # then we merge those values to the base_config.
        custom_config = load_config(config_file)
        config = merge_into(custom_config, config)

    if override_params:
        override_config = parse_override(override_params)
        config = merge_into(override_config, config)

    if config.train.debug or config.train.tf_debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    return config


def get_learning_rate(config, global_step=None):
    method = config.train.learning_rate_decay_method
    if not method or method == 'none':
        return config.train.initial_learning_rate

    if method == 'piecewise_constant':
        return tf.train.piecewise_constant(
            global_step, boundaries=[
                tf.cast(config.train.learning_rate_decay, tf.int64), ],
            values=[
                config.train.initial_learning_rate,
                config.train.initial_learning_rate * 0.1
            ], name='learning_rate_piecewise_constant'
        )

    if method == 'exponential_decay':
        return tf.train.exponential_decay(
            learning_rate=config.initial_learning_rate,
            global_step=global_step,
            decay_steps=config.train.learning_rate_decay, decay_rate=0.96,
            staircase=True, name='learning_rate_with_decay'
        )

    raise ValueError('Invalid learning_rate method "{}"'.format(method))


def run(target, cluster_spec, is_chief, model_type, config_file,
        override_params, continue_training, seed=0, **kwargs):

    if seed:
        tf.set_random_seed(seed)

    model_class = get_model(model_type)
    config = get_model_config(
        model_class, config_file, override_params, **kwargs)
    model = model_class(config, seed=seed)

    # Placement of ops on devices using replica device setter
    # which automatically places the parameters on the `ps` server
    # and the `ops` on the workers
    #
    # See:
    # https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter
    with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
        dataset = TFRecordDataset(config, seed=seed)
        train_dataset = dataset()

        train_image = train_dataset['image']
        train_filename = train_dataset['filename']
        train_bboxes = train_dataset['bboxes']

        # TODO: This is not the best place to configure rank? Why is rank not
        # transmitted through the queue
        train_image.set_shape((None, None, 3))
        # We add fake batch dimension to train data.
        # TODO: DEFINITELY NOT THE BEST PLACE
        train_image = tf.expand_dims(train_image, 0)

        prediction_dict = model(train_image, train_bboxes, training=True)
        total_loss = model.loss(prediction_dict)

        global_step = tf.contrib.framework.get_or_create_global_step()

        # load_weights returns no_op when empty checkpoint_file.
        # TODO: Make optional for different types of models.
        load_op = model.load_pretrained_weights()

        # TODO: what is this? probably broken since code changes for
        #       distributed training
        # saver = model.get_saver()
        # if config.train.ignore_scope:
        #     partial_loader = model.get_saver(
        #         ignore_scope=config.train.ignore_scope
        #     )

        learning_rate = get_learning_rate(config, global_step)
        tf.summary.scalar('losses/learning_rate', learning_rate)

        optimizer_cls = OPTIMIZERS[config.train.optimizer_type]
        if config.train.optimizer_type == 'momentum':
            optimizer = optimizer_cls(learning_rate, config.train.momentum)
        else:
            optimizer = optimizer_cls(learning_rate)

        trainable_vars = model.get_trainable_vars()

        # Compute, clip and apply gradients
        grads_and_vars = optimizer.compute_gradients(
            total_loss, trainable_vars
        )

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

        for grad, var in grads_and_vars:
            if grad is not None:
                variable_summaries(
                    grad, 'clipped_grad/{}'.format(var.name[:-2]))

        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step
        )

        # Create initializer for variables.
        init_op = tf.group(
            tf.global_variables_initializer(),
            # Queue-related variables need a special initializer
            tf.local_variables_initializer(),
            # Load pre-trained weights of part of the network only
            load_op
        )

    tf.logging.info('Starting training for {}'.format(model))

    run_options = None
    if config.train.full_trace:
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE
        )

    # Create custom Scaffold to make sure we run our own init_op when model
    # is not restored from checkpoint.
    scaffold = tf.train.Scaffold(init_op=init_op)

    #
    # Custom hooks for our session
    #
    hooks = []
    if config.train.tf_debug:
        debug_hook = tf_debug.LocalCLIDebugHook()
        debug_hook.add_tensor_filter(
            'has_inf_or_nan', tf_debug.has_inf_or_nan
        )
        hooks.extend([debug_hook])

    with tf.train.MonitoredTrainingSession(
        master=target,
        is_chief=is_chief,
        checkpoint_dir=config.train.job_dir,
        scaffold=scaffold,
        hooks=hooks,
        save_checkpoint_secs=config.train.save_checkpoint_secs,
        save_summaries_steps=config.train.save_summaries_steps,
        save_summaries_secs=config.train.save_summaries_secs,
    ) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                _, train_loss, step, filename = sess.run([
                    train_op, total_loss, global_step, train_filename
                ], options=run_options)

                # TODO: Add image summary when master.

                tf.logging.info('step: {}, file: {}, train_loss: {}'.format(
                    step, filename, train_loss
                ))

        except tf.errors.OutOfRangeError:
            tf.logging.info(
                'finished training after {} epoch limit'.format(
                    config.train.num_epochs
                )
            )

            # TODO: Print summary
        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)


@click.command(help='Train models')
@click.option('model_type', '--model', required=True, default='fasterrcnn')  # noqa
@click.option('config_file', '--config', '-c', help='Config to use.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
@click.option('--continue-training', is_flag=True, help='Continue training using model dir and run name.')  # noqa
@click.option('--seed', type=float, help='Global seed value for random operations.')  # noqa
@click.option('--checkpoint-file', help='Weight checkpoint to resuming training from.')  # noqa
@click.option('--ignore-scope', help='Used to ignore variables when loading from checkpoint.')  # noqa
@click.option('--log-dir', default='/tmp/luminoth/', help='Directory for Tensorboard logs.')  # noqa
@click.option('--tf-debug', is_flag=True, help='Create debugging Tensorflow session with tfdb.')  # noqa
@click.option('--debug', is_flag=True, help='Debug mode (DEBUG log level and intermediate variables are returned)')  # noqa
@click.option('--no-log/--log', default=False, help='Save or don\'t summary logs.')  # noqa
@click.option('--display-every', default=500, type=int, help='Show image debug information every N batches (debug mode must be activated)')  # noqa
@click.option('--random-shuffle/--fifo', default=True, help='Ingest data from dataset in random order.')  # noqa
@click.option('--save-timeline', is_flag=True, help='Save timeline of execution (debug mode must be activated).')  # noqa
@click.option('--full-trace', is_flag=True, help='Run graph session with FULL_TRACE config (for memory and running time debugging)')  # noqa
@click.option('--initial-learning-rate', default=0.0001, type=float, help='Initial learning rate.')  # noqa
@click.option('--learning-rate-decay', default=10000, type=int, help='Decay learning rate after N batches.')  # noqa
@click.option('--learning-rate-decay-method', default='piecewise_constant', type=click.Choice(LEARNING_RATE_DECAY_METHODS), help='Tipo of learning rate decay to use.')  # noqa
@click.option('optimizer_type', '--optimizer', default='momentum', type=click.Choice(OPTIMIZERS.keys()), help='Optimizer to use.')  # noqa
@click.option('--momentum', default=0.9, type=float, help='Momentum to use when using the MomentumOptimizer.')  # noqa
@click.option('--job-dir')
def train(*args, **kwargs):
    """
    Parse TF_CONFIG to cluster_spec and call run_train() method.

    TF_CONFIG environment variable is available when running using
    gcloud either locally or on cloud. It has all the information required
    to create a ClusterSpec which is important for running distributed code.
    """
    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available, run local
    if not tf_config:
        return run('', None, True, *args, **kwargs)

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return run('', None, True, *args, **kwargs)

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(
        cluster_spec, job_name=job_name, task_index=task_index)

    # Wait for incoming connections forever
    # Worker ships the graph to the ps server
    # The ps server manages the parameters of the model.
    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        is_master = job_name == 'master'
        return run(server.target, cluster_spec, is_master, *args, **kwargs)


if __name__ == '__main__':
    train()
