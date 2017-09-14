import click
import os
import tensorflow as tf
import time

from luminoth.datasets import TFRecordDataset
from luminoth.models import get_model
from luminoth.utils.config import (
    load_config, merge_into, kwargs_to_config, parse_override
)

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'momentum': tf.train.MomentumOptimizer,
}

LEARNING_RATE_DECAY_METHODS = set([
    'piecewise_constant', 'exponential_decay', 'none'
])


@click.command(help='Train models')
@click.option('model_type', '--model', required=True, default='fasterrcnn')  # noqa
@click.option('config_file', '--config', '-c', help='Config to use.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
@click.option('--continue-training', is_flag=True, help='Continue training using model dir and run name.')  # noqa
@click.option('--model-dir', default='models/', help='Directory to save the partial trained models.')  # noqa
@click.option('--checkpoint-file', help='Weight checkpoint to resuming training from.')  # noqa
@click.option('--ignore-scope', help='Used to ignore variables when loading from checkpoint.')  # noqa
@click.option('--log-dir', default='/tmp/luminoth/', help='Directory for Tensorboard logs.')  # noqa
@click.option('--save-every', default=1000, help='Save checkpoint after that many batches.')  # noqa
@click.option('--tf-debug', is_flag=True, help='Create debugging Tensorflow session with tfdb.')  # noqa
@click.option('--debug', is_flag=True, help='Debug mode (DEBUG log level and intermediate variables are returned)')  # noqa
@click.option('--run-name', default='train', help='Run name used to log in Tensorboard and isolate checkpoints.')  # noqa
@click.option('--no-log/--log', default=False, help='Save or don\'t summary logs.')  # noqa
@click.option('--display-every', default=500, type=int, help='Show image debug information every N batches (debug mode must be activated)')  # noqa
@click.option('--random-shuffle/--fifo', default=True, help='Ingest data from dataset in random order.')  # noqa
@click.option('--save-timeline', is_flag=True, help='Save timeline of execution (debug mode must be activated).')  # noqa
@click.option('--summary-every', default=1, type=int, help='Save summary logs every N batches.')  # noqa
@click.option('--full-trace', is_flag=True, help='Run graph session with FULL_TRACE config (for memory and running time debugging)')  # noqa
@click.option('--initial-learning-rate', default=0.0001, type=float, help='Initial learning rate.')  # noqa
@click.option('--learning-rate-decay', default=10000, type=int, help='Decay learning rate after N batches.')  # noqa
@click.option('--learning-rate-decay-method', default='piecewise_constant', type=click.Choice(LEARNING_RATE_DECAY_METHODS), help='Tipo of learning rate decay to use.')  # noqa
@click.option('optimizer_type', '--optimizer', default='momentum', type=click.Choice(OPTIMIZERS.keys()), help='Optimizer to use.')  # noqa
@click.option('--momentum', default=0.9, type=float, help='Momentum to use when using the MomentumOptimizer.')  # noqa
@click.option('--job-dir')  # TODO: Ignore this arg passed by Google Cloud ML.
def train(model_type, config_file, override_params, continue_training,
          **kwargs):

    model_class = get_model(model_type)
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

    model = model_class(config)
    dataset = TFRecordDataset(config)
    train_dataset = dataset()

    train_image = train_dataset['image']
    train_filename = train_dataset['filename']

    train_bboxes = train_dataset['bboxes']

    # TODO: This is not the best place to configure rank? Why is rank not
    # transmitted through the queue
    train_image.set_shape((None, None, 3))
    # We add fake batch dimension to train data. TODO: DEFINITELY NOT THE BEST
    # PLACE
    train_image = tf.expand_dims(train_image, 0)

    prediction_dict = model(train_image, train_bboxes, training=True)

    total_loss = model.loss(prediction_dict)

    initial_global_step = 0
    checkpoint_path = os.path.join(
        config.train.model_dir, config.train.run_name,
        model.scope_name
    )
    if continue_training:
        last_checkpoint = tf.train.get_checkpoint_state(
            os.path.dirname(checkpoint_path)
        )
        if not last_checkpoint or not last_checkpoint.model_checkpoint_path:
            raise ValueError(
                'Could not find checkpoint in {}. Check run name'.format(
                    checkpoint_path))
        initial_global_step = int(
            last_checkpoint.model_checkpoint_path.split('-')[-1]
        )
        tf.logging.info('Starting training from global_step {}'.format(
            initial_global_step))
        last_checkpoint_path = last_checkpoint.model_checkpoint_path
        tf.logging.info('Using checkpoint "{}"'.format(last_checkpoint_path))
        config.train.checkpoint_file = last_checkpoint_path

    global_step_init = tf.constant_initializer(initial_global_step)

    global_step = tf.get_variable(
        name="global_step", shape=[], dtype=tf.int64,
        initializer=global_step_init, trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP]
    )

    # load_weights returns no_op when empty checkpoint_file.
    # TODO: Make optional for different types of models.
    load_op = model.load_pretrained_weights()

    saver = model.get_saver()
    if config.train.ignore_scope:
        partial_loader = model.get_saver(
            ignore_scope=config.train.ignore_scope
        )

    learning_rate_decay_method = config.train.learning_rate_decay_method
    if not learning_rate_decay_method or learning_rate_decay_method == 'none':
        learning_rate = config.train.initial_learning_rate
    elif learning_rate_decay_method == 'piecewise_constant':
        learning_rate = tf.train.piecewise_constant(
            global_step, boundaries=[
                tf.cast(config.train.learning_rate_decay, tf.int64), ],
            values=[
                config.train.initial_learning_rate,
                config.train.initial_learning_rate * 0.1
            ], name='learning_rate_piecewise_constant'
        )
    elif learning_rate_decay_method == 'exponential_decay':
        learning_rate = tf.train.exponential_decay(
            learning_rate=config.initial_learning_rate,
            global_step=global_step,
            decay_steps=config.train.learning_rate_decay, decay_rate=0.96,
            staircase=True, name='learning_rate_with_decay'
        )
    else:
        raise ValueError(
            'Invalid learning_rate method "{}"'.format(
                learning_rate_decay_method))

    tf.summary.scalar('losses/learning_rate', learning_rate)

    optimizer_cls = OPTIMIZERS[config.train.optimizer_type]
    if config.train.optimizer_type == 'momentum':
        optimizer = optimizer_cls(learning_rate, config.train.momentum)
    else:
        optimizer = optimizer_cls(learning_rate)

    trainable_vars = model.get_trainable_vars()
    grads_and_vars = optimizer.compute_gradients(total_loss, trainable_vars)

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

    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step
    )

    # Create initializer for variables. Queue-related variables need a special
    # initializer.
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    # TODO: Why do we need to run this?
    metric_ops = tf.get_collection('metric_ops')
    # metrics = tf.get_collection('metrics')

    tf.logging.info('Starting training for {}'.format(model))

    summary_dir = os.path.join(config.train.log_dir, config.train.run_name)
    summarizer = tf.summary.merge([
        tf.summary.merge_all(),
        model.summary,
    ])

    with tf.Session() as sess:

        sess.run(init_op)  # initialized variables
        sess.run(load_op)  # load pretrained weights

        # Restore all variables from checkpoint file
        if config.train.checkpoint_file:
            # If ignore_scope is set, we don't load those variables from
            # checkpoint.
            if config.train.ignore_scope:
                partial_loader.restore(sess, config.train.checkpoint_file)
            else:
                saver.restore(sess, config.train.checkpoint_file)

        if not config.train.no_log:
            writer = tf.summary.FileWriter(summary_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = initial_global_step

        if config.train.tf_debug:
            from tensorflow.python import debug as tensorflow_debug
            sess = tensorflow_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter(
                'has_inf_or_nan', tensorflow_debug.has_inf_or_nan
            )

        try:
            while not coord.should_stop():
                write_summary = (
                    not config.train.no_log and
                    (step + 1) % config.train.summary_every == 0
                )

                run_metadata = None
                if write_summary:
                    run_metadata = tf.RunMetadata()

                run_options = None
                if config.train.full_trace:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE
                    )

                display_images = (
                    config.train.debug and
                    (step + 1) % config.train.display_every == 0
                )

                fetches = {
                    'train': train_op,
                    'train_loss': total_loss,
                    'step': global_step,
                    'filename': train_filename,
                }

                if display_images:
                    fetches['prediction_dict'] = prediction_dict

                if write_summary:
                    fetches['summary'] = summarizer
                    fetches['metrics'] = metric_ops

                before = time.time()
                fetched = sess.run(
                    fetches, run_metadata=run_metadata, options=run_options
                )

                train_loss = fetched['train_loss']
                step = fetched['step']
                filename = fetched['filename']

                tf.logging.info(
                    'step: {}, file: {}, train_loss: {} (in {:.2f}s)'.format(
                        step, filename, train_loss, time.time() - before
                    )
                )

                if write_summary:
                    before = time.time()
                    summary = fetched['summary']
                    writer.add_summary(summary, step)
                    if step == 1:
                        # only add the run_metadata for first step.
                        writer.add_run_metadata(
                            run_metadata, str(step)
                        )
                    tf.logging.info('wrote summary in {:.2f}s'.format(
                        time.time() - before
                    ))

                if display_images:
                    before = time.time()
                    from luminoth.utils.image_vis import (
                        add_images_to_tensoboard
                    )
                    pred_dict = fetched['prediction_dict']
                    add_images_to_tensoboard(
                        pred_dict, step, summary_dir, config.network.with_rcnn
                    )
                    tf.logging.info(
                        'saved images in summary in {:.2f}s'.format(
                            time.time() - before
                        )
                    )

                    if config.train.save_timeline:
                        from tensorflow.python.client import timeline
                        run_tmln = timeline.Timeline(
                            run_metadata.step_stats)
                        chrome_trace = run_tmln.generate_chrome_trace_format()
                        timeline_filename = 'timeline_{}.json'.format(step)
                        with tf.gfile.GFile(timeline_filename, 'w') as f:
                            f.write(chrome_trace)

                if not config.train.no_log:
                    if step % config.train.save_every == 0:
                        # We don't support partial saver.
                        before = time.time()
                        saver.save(sess, checkpoint_path, global_step=step)
                        tf.logging.info('saving checkpoint in {:.2f}s'.format(
                            time.time() - before
                        ))

        except tf.errors.OutOfRangeError:
            tf.logging.info('step = {}, train_loss = {:.2f}'.format(
                step, train_loss))
            tf.logging.info('finished training -- epoch limit reached')
            # TODO: Print summary
        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)


if __name__ == '__main__':
    train()
