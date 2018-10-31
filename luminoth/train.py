import click
import json
import os
import sys
import tensorflow as tf
import time

from tensorflow.python import debug as tf_debug

from luminoth.datasets import get_dataset
from luminoth.datasets.exceptions import InvalidDataDirectory
from luminoth.models import get_model
from luminoth.utils.config import get_config
from luminoth.utils.hooks import ImageVisHook, VarVisHook
from luminoth.utils.training import get_optimizer, clip_gradients_by_norm
from luminoth.utils.experiments import save_run


def run(config, target='', cluster_spec=None, is_chief=True, job_name=None,
        task_index=None, get_model_fn=get_model, get_dataset_fn=get_dataset,
        environment=None):
    model_class = get_model_fn(config.model.type)

    image_vis = config.train.get('image_vis')
    var_vis = config.train.get('var_vis')

    if config.train.get('seed') is not None:
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
        try:
            config['dataset']['type']
        except KeyError:
            raise KeyError('dataset.type should be set on the custom config.')

        try:
            dataset_class = get_dataset_fn(config.dataset.type)
            dataset = dataset_class(config)
            train_dataset = dataset()
        except InvalidDataDirectory as exc:
            tf.logging.error(
                "Error while reading dataset, {}".format(exc)
            )
            sys.exit(1)

        train_image = train_dataset['image']
        train_filename = train_dataset['filename']
        train_bboxes = train_dataset['bboxes']

        prediction_dict = model(train_image, train_bboxes, is_training=True)
        total_loss = model.loss(prediction_dict)

        global_step = tf.train.get_or_create_global_step()

        optimizer = get_optimizer(config.train, global_step)

        # TODO: Is this necesarry? Couldn't we just get them from the
        # trainable vars collection? We should probably improve our
        # usage of collections.
        trainable_vars = model.get_trainable_vars()

        # Compute, clip and apply gradients
        with tf.name_scope('gradients'):
            grads_and_vars = optimizer.compute_gradients(
                total_loss, trainable_vars
            )

            if config.train.clip_by_norm:
                grads_and_vars = clip_gradients_by_norm(grads_and_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step
            )

        # Create custom init for slots in optimizer, as we don't save them to
        # our checkpoints. An example of slots in an optimizer are the Momentum
        # variables in MomentumOptimizer. We do this because slot variables can
        # effectively duplicate the size of your checkpoint!
        slot_variables = [
            optimizer.get_slot(var, name)
            for name in optimizer.get_slot_names()
            for var in trainable_vars
        ]
        slot_init = tf.variables_initializer(
            slot_variables,
            name='optimizer_slots_initializer'
        )

        # Create saver for saving/restoring model
        model_saver = tf.train.Saver(
            set(tf.global_variables()) - set(slot_variables),
            name='model_saver',
            max_to_keep=config.train.get('checkpoints_max_keep', 1),
        )

        # Create saver for loading pretrained checkpoint into base network
        base_checkpoint_vars = model.get_base_network_checkpoint_vars()
        checkpoint_file = model.get_checkpoint_file()
        if base_checkpoint_vars and checkpoint_file:
            base_net_checkpoint_saver = tf.train.Saver(
                base_checkpoint_vars,
                name='base_net_checkpoint_saver'
            )

            # We'll send this fn to Scaffold init_fn
            def load_base_net_checkpoint(_, session):
                base_net_checkpoint_saver.restore(
                    session, checkpoint_file
                )
        else:
            load_base_net_checkpoint = None

    tf.logging.info('{}Starting training for {}'.format(log_prefix, model))

    run_options = None
    if config.train.full_trace:
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE
        )

    # Create custom Scaffold to make sure we run our own init_op when model
    # is not restored from checkpoint.
    summary_op = [model.summary]
    summaries = tf.summary.merge_all()
    if summaries is not None:
        summary_op.append(summaries)
    summary_op = tf.summary.merge(summary_op)

    # `ready_for_local_init_op` is hardcoded to 'ready' as local init doesn't
    # depend on global init and `local_init_op` only runs when it is set as
    # 'ready' (an empty string tensor sets it as ready).
    scaffold = tf.train.Scaffold(
        saver=model_saver,
        init_op=tf.global_variables_initializer() if is_chief else tf.no_op(),
        local_init_op=tf.group(tf.initialize_local_variables(), slot_init),
        ready_for_local_init_op=tf.constant([], dtype=tf.string),
        summary_op=summary_op,
        init_fn=load_base_net_checkpoint,
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
        checkpoint_dir = None
    elif config.train.run_name:
        # Use run_name when available
        checkpoint_dir = os.path.join(
            config.train.job_dir, config.train.run_name
        )
    else:
        checkpoint_dir = config.train.job_dir

    should_add_hooks = (
        config.train.display_every_steps
        or config.train.display_every_secs
        and checkpoint_dir is not None
    )
    if should_add_hooks:
        if not config.train.debug and image_vis == 'debug':
            tf.logging.warning('ImageVisHook will not run without debug mode.')
        elif image_vis is not None:
            # ImageVis only runs on the chief.
            chief_only_hooks.append(
                ImageVisHook(
                    prediction_dict,
                    image=train_dataset['image'],
                    gt_bboxes=train_dataset['bboxes'],
                    config=config.model,
                    output_dir=checkpoint_dir,
                    every_n_steps=config.train.display_every_steps,
                    every_n_secs=config.train.display_every_secs,
                    image_visualization_mode=image_vis
                )
            )

        if var_vis is not None:
            # VarVis only runs on the chief.
            chief_only_hooks.append(
                VarVisHook(
                    every_n_steps=config.train.display_every_steps,
                    every_n_secs=config.train.display_every_secs,
                    mode=var_vis,
                    output_dir=checkpoint_dir,
                    vars_summary=model.vars_summary,
                )
            )

    step = -1
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

                if is_chief and step == 1:
                    # We save the run after first batch to make sure everything
                    # works properly.
                    save_run(config, environment=environment)

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

        return step


@click.command(help='Train models')
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Config to use.')  # noqa
@click.option('--job-dir', help='Job directory.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
def train(config_files, job_dir, override_params):
    """
    Parse TF_CONFIG to cluster_spec and call run() function
    """
    # TF_CONFIG environment variable is available when running using gcloud
    # either locally or on cloud. It has all the information required to create
    # a ClusterSpec which is important for running distributed code.
    tf_config_val = os.environ.get('TF_CONFIG')

    if tf_config_val:
        tf_config = json.loads(tf_config_val)
    else:
        tf_config = {}

    cluster = tf_config.get('cluster')
    job_name = tf_config.get('task', {}).get('type')
    task_index = tf_config.get('task', {}).get('index')
    environment = tf_config.get('environment', 'local')

    # Get the user config and the model type from it.
    try:
        config = get_config(config_files, override_params=override_params)
    except KeyError:
        # Without mode type defined we can't use the default config settings.
        raise KeyError('model.type should be set on the custom config.')

    if job_dir:
        override_params += ('train.job_dir={}'.format(job_dir), )

    # If cluster information is empty or TF_CONFIG is not available, run local
    if job_name is None or task_index is None:
        return run(
            config, environment=environment
        )

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
            config, target=server.target, cluster_spec=cluster_spec,
            is_chief=is_chief, job_name=job_name, task_index=task_index,
            environment=environment
        )


if __name__ == '__main__':
    train()
