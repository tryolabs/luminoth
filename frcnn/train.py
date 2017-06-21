import numpy as np
import os
import sonnet as snt
import tensorflow as tf
import click

from .network import FasterRCNN
from .config import Config
from .dataset import TFRecordDataset


@click.command()
@click.option('--num-classes', default=20)
@click.option('--pretrained-net', default='vgg_16', type=click.Choice(['vgg_16']))
@click.option('--pretrained-weights')
@click.option('--model-dir', default='models/')
@click.option('--checkpoint-file')
@click.option('--log_dir', default='/tmp/frcnn/')
@click.option('--save_every', default=10)
@click.option('--debug', is_flag=True)
def train(num_classes, pretrained_net, pretrained_weights, model_dir, checkpoint_file, log_dir, save_every, debug):

    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    model = FasterRCNN(Config, num_classes=num_classes)
    dataset = TFRecordDataset(Config, num_classes=num_classes)
    train_dataset = dataset()

    train_image = train_dataset['image']
    train_filename = train_dataset['filename']
    # TODO: This is not the best place to configure rank? Why is rank not
    # transmitted through the queue
    train_image.set_shape((None, None, 3))

    train_bboxes = train_dataset['bboxes']

    # We add fake batch dimension to train data. TODO: DEFINITELY NOT THE BEST
    # PLACE
    train_image = tf.expand_dims(train_image, 0)
    # Bbox doesn't need a dimension for batch TODO: Necesitamos standarizar esto!
    # train_bboxes = tf.expand_dims(train_bboxes, 0)

    prediction_dict = model(train_image, train_bboxes)

    saver = snt.get_saver(model)
    summarizer = tf.summary.merge_all()

    if pretrained_weights:
        # TODO: Calling _pretrained _load_weights sucks. We need better abstraction
        # Maybe handle it inside the model?
        # TODO: Prefixes should be known by the model?
        load_op = model._pretrained._load_weights(checkpoint_file=pretrained_weights, old_prefix='vgg_16/', new_prefix='fasterrcnn/vgg/')
    else:
        load_op = tf.no_op(name='not_loading_pretrained')

    total_loss = model.loss(prediction_dict)

    tf.summary.scalar('loss', total_loss)

    initial_learning_rate = 0.001

    learning_rate = tf.get_variable(
        "learning_rate",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(initial_learning_rate),
        trainable=False)

    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # TODO: We should define `var_list`
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    # Create initializer for variables. Queue-related variables need a special
    # initializer.
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    metric_ops = tf.get_collection('metric_ops')
    metrics = tf.get_collection('metrics')

    tf.logging.info('Starting training for {}'.format(model))

    with tf.Session() as sess:
        sess.run(init_op)  # initialized variables
        sess.run(load_op)  # load pretrained weights

        # Restore all variables from checkpoint file
        if checkpoint_file:
            saver.restore(sess, checkpoint_file)

        writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'train'), sess.graph
        )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        count_images = 0

        if debug:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        try:
            while not coord.should_stop():

                run_metadata = tf.RunMetadata()

                _, summary, train_loss, step, pred_dict, filename, *_ = sess.run([
                train_op, summarizer, total_loss, global_step, prediction_dict, train_filename, metric_ops
                ], run_metadata=run_metadata)

                count_images += 1

                tf.logging.info('train_loss: {}'.format(train_loss))
                tf.logging.info('step: {}, filename: {}'.format(step, filename))

                values = sess.run(metrics)

                if step % save_every == 0:
                    saver.save(sess, os.path.join(model_dir, model.scope_name), global_step=step)

                writer.add_summary(summary, step)
                writer.add_run_metadata(run_metadata, 'step{}'.format(step))

        except tf.errors.OutOfRangeError:
            tf.logging.info('iter = {}, train_loss = {:.2f}'.format(step, train_loss))
            tf.logging.info('finished training -- epoch limit reached')
            tf.logging.info('count_images = {}'.format(count_images))
        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)


if __name__ == '__main__':
    train()
