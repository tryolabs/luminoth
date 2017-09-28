import os

import click
import json
import tensorflow as tf
import time

from tensorflow.python import debug as tf_debug

from luminoth.datasets import TFRecordDataset
from luminoth.models import get_model
from luminoth.utils.config import get_model_config
from luminoth.utils.hooks import ImageVisHook
from luminoth.utils.training import (
    get_optimizer, clip_gradients_by_norm
)


def run(model_type, config_file, override_params, target='', cluster_spec=None,
        is_chief=True, job_name=None, task_index=None, **kwargs):

    model_class = get_model(model_type)

    config = get_model_config(
        model_class.base_config, config_file, override_params, **kwargs
    )

    if config.train.seed is not None:
        tf.set_random_seed(config.train.seed)

    log_prefix = '[{}-{}] - '.format(job_name, task_index) \
        if job_name is not None and task_index is not None else ''

    if config.train.debug or config.train.tf_debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    model = model_class(config)

    # Placement of ops on devices using replica device setter
    # which automatically places the parameters on the `ps` server
    # and the `ops` on the workers
    #
    # See:
    # https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter
    with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
        dataset = TFRecordDataset(config)
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

        prediction_dict = model(train_image, train_bboxes, is_training=True)
        total_loss = model.loss(prediction_dict)

        global_step = tf.contrib.framework.get_or_create_global_step()

        optimizer = get_optimizer(config.train, global_step)

        trainable_vars = model.get_trainable_vars()

        # Compute, clip and apply gradients
        grads_and_vars = optimizer.compute_gradients(
            total_loss, trainable_vars
        )

        # Clip by norm. TODO: Configurable
        grads_and_vars = clip_gradients_by_norm(grads_and_vars)

        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step
        )

    tf.logging.info('{}Starting training for {}'.format(log_prefix, model))

    run_options = None
    if config.train.full_trace:
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE
        )

    if is_chief:
        # Load pretrained weights needs to be called before defining the train
        # op. After it, variables for the optimizer are created.
        with tf.control_dependencies([tf.global_variables_initializer()]):
            with tf.control_dependencies([model.load_pretrained_weights()]):
                init_op = tf.no_op(name='global_init_load_pretrained')
    else:
        init_op = tf.no_op()

    # Create custom Scaffold to make sure we run our own init_op when model
    # is not restored from checkpoint.
    scaffold = tf.train.Scaffold(
        # Initialize local and global variables.
        init_op=init_op,
        # Queue-related variables need a special initializer.
        local_init_op=tf.local_variables_initializer(),
        summary_op=tf.summary.merge([
            tf.summary.merge_all(),
            model.summary,
        ])
    )

    # Custom hooks for our session
    hooks = []
    chief_only_hooks = []

    if config.train.tf_debug:
        debug_hook = tf_debug.LocalCLIDebugHook()
        debug_hook.add_tensor_filter(
            'has_inf_or_nan', tf_debug.has_inf_or_nan
        )
        hooks.extend([debug_hook])

    if not config.train.job_dir:
        tf.logging.warning(
            '`job_dir` is not defined. Checkpoints and logs will not be saved.'
        )
    elif config.train.run_name:
        # Use run_name when available
        checkpoint_dir = os.path.join(
            config.train.job_dir, config.train.run_name
        )
    else:
        checkpoint_dir = config.train.job_dir

    if config.train.display_every_steps or config.train.display_every_secs:
        if not config.train.debug:
            tf.logging.warning('ImageVisHook will not run without debug mode.')
        else:
            # ImageVis only runs on the chief.
            chief_only_hooks.append(
                ImageVisHook(
                    prediction_dict, with_rcnn=config.network.with_rcnn,
                    output_dir=checkpoint_dir,
                    every_n_steps=config.train.display_every_steps,
                    every_n_secs=config.train.display_every_secs
                )
            )

    with tf.train.MonitoredTrainingSession(
        master=target,
        is_chief=is_chief,
        checkpoint_dir=checkpoint_dir,
        scaffold=scaffold,
        hooks=hooks,
        chief_only_hooks=chief_only_hooks,
        save_checkpoint_secs=config.train.save_checkpoint_secs,
        save_summaries_steps=config.train.save_summaries_steps,
        save_summaries_secs=config.train.save_summaries_secs,
    ) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                before = time.time()
                _, train_loss, step, filename = sess.run([
                    train_op, total_loss, global_step, train_filename
                ], options=run_options)

                # TODO: Add image summary every once in a while.

                tf.logging.info(
                    '{}step: {}, file: {}, train_loss: {}, in {:.2f}s'.format(
                        log_prefix, step, filename, train_loss,
                        time.time() - before
                    ))

        except tf.errors.OutOfRangeError:
            tf.logging.info(
                '{}finished training after {} epoch limit'.format(
                    log_prefix, config.train.num_epochs
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
@click.option('--seed', type=float, help='Global seed value for random operations.')  # noqa
@click.option('--checkpoint-file', help='Weight checkpoint to resuming training from.')  # noqa
@click.option('--ignore-scope', help='Used to ignore variables when loading from checkpoint.')  # noqa
@click.option('--tf-debug', is_flag=True, help='Create debugging Tensorflow session with tfdb.')  # noqa
@click.option('--debug', is_flag=True, help='Debug mode (DEBUG log level and intermediate variables are returned)')  # noqa
@click.option('--run-name', type=str, help='Run name used to log in Tensorboard and isolate checkpoints.')  # noqa
@click.option('--no-log/--log', default=False, help='Save or don\'t summary logs.')  # noqa
@click.option('--random-shuffle/--fifo', default=True, help='Ingest data from dataset in random order.')  # noqa
@click.option('--save-timeline', is_flag=True, help='Save timeline of execution (debug mode must be activated).')  # noqa
@click.option('--full-trace', is_flag=True, help='Run graph session with FULL_TRACE config (for memory and running time debugging)')  # noqa
@click.option('--job-dir', type=str, help='Directory where to save logs and checkpoints from training.')  # noqa
def train(*args, **kwargs):
    """
    Parse TF_CONFIG to cluster_spec and call run() function
    """

    # TF_CONFIG environment variable is available when running using
    # gcloud either locally or on cloud. It has all the information required
    # to create a ClusterSpec which is important for running distributed code.
    tf_config_val = os.environ.get('TF_CONFIG')

    if tf_config_val:
        tf_config = json.loads(tf_config_val)
    else:
        tf_config = {}

    cluster = tf_config.get('cluster')
    job_name = tf_config.get('task', {}).get('type')
    task_index = tf_config.get('task', {}).get('index')

    # If cluster information is empty or TF_CONFIG is not available, run local
    if job_name is None or task_index is None:
        return run(*args, **kwargs)

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
        is_chief = job_name == 'master'
        return run(
            *args,
            target=server.target, cluster_spec=cluster_spec,
            is_chief=is_chief, job_name=job_name, task_index=task_index,
            **kwargs
        )


if __name__ == '__main__':
    train()
