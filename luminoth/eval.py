import click
import tensorflow as tf
import os

from .dataset import TFRecordDataset
from .models import MODELS, PRETRAINED_MODELS
from .utils.config import (
    load_config, merge_into, parse_override
)
from .utils.vars import get_saver


@click.command(help='Evaluate trained (or training) models')
@click.argument('model-type', type=click.Choice(MODELS.keys()))
@click.argument('dataset-split', default='val')
@click.option('config_file', '--config', '-c', type=click.File('r'), help='Config to use.')
@click.option('--model-dir', required=True, help='Directory from where to read saved models.')
@click.option('--log-dir', help='Directory where to save evaluation logs.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')
def evaluate(model_type, dataset_split, config_file, model_dir, log_dir,
             override_params):
    """
    Evaluate models using dataset.
    """
    model_class = MODELS[model_type.lower()]
    config = model_class.base_config

    if config_file:
        # If we have a custom config file overwritting default settings
        # then we merge those values to the base_config.
        custom_config = load_config(config_file)
        config = merge_into(custom_config, config)

    config.train.model_dir = model_dir or config.train.model_dir
    config.train.log_dir = log_dir or config.train.log_dir

    if override_params:
        override_config = parse_override(override_params)
        config = merge_into(override_config, config)

    # Override default dataset split
    config.dataset.split = dataset_split

    model = model_class(config)
    pretrained = PRETRAINED_MODELS[config.pretrained.net](
        config.pretrained
    )
    dataset = TFRecordDataset(config)
    train_dataset = dataset()

    train_image = train_dataset['image']
    # train_filename = train_dataset['filename']

    train_bboxes = train_dataset['bboxes']

    # TODO: This is not the best place to configure rank? Why is rank not
    # transmitted through the queue
    train_image.set_shape((None, None, 3))
    # We add fake batch dimension to train data. TODO: DEFINITELY NOT THE BEST
    # PLACE
    train_image = tf.expand_dims(train_image, 0)

    pretrained_dict = pretrained(train_image, is_training=False)
    prediction_dict = model(
        train_image, pretrained_dict['net'], train_bboxes, is_training=False
    )

    pred_objects = prediction_dict['classification_prediction']['objects']
    pred_objects_labels = prediction_dict['classification_prediction']['objects_labels']

    # metrics(pred_objects, pred_objects_labels, train_bboxes)

    # batch_loss = model.loss(prediction_dict)
    # total_loss, _ = tf.metrics.mean(
    #     batch_loss, name='loss',
    #     metrics_collections='metrics',
    #     updates_collections='metric_ops',
    # )
    # tf.summary.scalar('loss', total_loss)

    metric_ops = tf.get_collection('metric_ops')
    metrics = tf.get_collection('metrics')

    summarizer = tf.summary.merge([
        tf.summary.merge_all(),
        model.summary,
    ])

    last_checkpoint = tf.train.get_checkpoint_state(
        model_dir
    )
    if not last_checkpoint or not last_checkpoint.model_checkpoint_path:
        raise ValueError(
            'Could not find checkpoint in {}. Check run name'.format(
                model_dir))

    config.train.run_name = os.path.split(
        os.path.dirname(last_checkpoint.model_checkpoint_path))[-1]

    global_step = int(
        last_checkpoint.model_checkpoint_path.split('-')[-1]
    )
    tf.logging.info('Evaluating global_step {}'.format(
        global_step))
    last_checkpoint_path = last_checkpoint.model_checkpoint_path
    tf.logging.info('Using checkpoint "{}"'.format(last_checkpoint_path))
    config.train.checkpoint_file = last_checkpoint_path

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    saver = get_saver((model, pretrained, ))

    # TODO: Get runname from model-dir
    summary_dir = os.path.join(config.train.log_dir, config.train.run_name)

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, config.train.checkpoint_file)

        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                summary, _ = sess.run([summarizer, metric_ops])
                values = sess.run(metrics)
                print('{} {}'.format(values, summary))

        except tf.errors.OutOfRangeError:
            writer.add_summary(summary, global_step)

        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)

