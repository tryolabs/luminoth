import numpy as np
import os
import sonnet as snt
import tensorflow as tf
import click

from .network import FasterRCNN
from .config import Config
from .dataset import TFRecordDataset
from .utils.image_vis import (
    draw_top_nms_proposals, draw_batch_proposals, draw_rpn_cls_loss,
    draw_rpn_bbox_pred, draw_rpn_bbox_pred_with_target, draw_positive_anchors,
    draw_anchors, draw_object_prediction
)


@click.command()
@click.option('--num-classes', default=20)
@click.option('--pretrained-net', default='vgg_16', type=click.Choice(['vgg_16']))
@click.option('--pretrained-weights')
@click.option('--model-dir', default='models/')
@click.option('--checkpoint-file')
@click.option('--ignore-scope')
@click.option('--log-dir', default='/tmp/frcnn/')
@click.option('--save-every', default=10)
@click.option('--debug', is_flag=True)
@click.option('--run-name', default='train')
@click.option('--with-rcnn', default=True, type=bool)
@click.option('--no-log', is_flag=True)
def train(num_classes, pretrained_net, pretrained_weights, model_dir,
          checkpoint_file, ignore_scope, log_dir, save_every, debug, run_name,
          with_rcnn, no_log):

    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    model = FasterRCNN(Config, num_classes=num_classes, with_rcnn=with_rcnn)
    dataset = TFRecordDataset(Config, num_classes=num_classes)
    train_dataset = dataset()

    train_image = train_dataset['image']
    train_filename = train_dataset['filename']
    train_scale_factor = train_dataset['scale_factor']
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

    model_variables = snt.get_normalized_variable_map(model, tf.GraphKeys.GLOBAL_VARIABLES)
    if ignore_scope:
        total_model_variables = len(model_variables)
        model_variables = {
            k: v for k, v in model_variables.items() if ignore_scope not in k
        }
        new_total_model_variables = len(model_variables)
        tf.logging.info('Not loading/saving {} variables with scope "{}"'.format(
            total_model_variables - new_total_model_variables, ignore_scope))

        partial_saver = tf.train.Saver(var_list=model_variables)

    saver = snt.get_saver(model)

    if pretrained_weights:
        # TODO: Calling _pretrained _load_weights sucks. We need better abstraction
        # Maybe handle it inside the model?
        # TODO: Prefixes should be known by the model?
        load_op = model._pretrained._load_weights(checkpoint_file=pretrained_weights, old_prefix='vgg_16/', new_prefix='fasterrcnn/vgg/')
    else:
        load_op = tf.no_op(name='not_loading_pretrained')

    total_loss = model.loss(prediction_dict)

    initial_learning_rate = 0.0001

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

    optimizer = tf.train.AdamOptimizer()  # TODO: parameter tunning

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

    summarizer = tf.summary.merge([
        tf.summary.merge_all(),
        model.summary,
    ])

    with tf.Session() as sess:
        sess.run(init_op)  # initialized variables
        sess.run(load_op)  # load pretrained weights

        # Restore all variables from checkpoint file
        if checkpoint_file:
            # TODO: We are better than this.

            # If ignore_scope is set, we don't load those variables from checkpoint.
            if ignore_scope:
                partial_saver.restore(sess, checkpoint_file)
            else:
                saver.restore(sess, checkpoint_file)

        if not no_log:
            writer = tf.summary.FileWriter(
                os.path.join(log_dir, run_name), sess.graph
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

                _, summary, train_loss, step, pred_dict, filename, scale_factor, *_ = sess.run([
                    train_op, summarizer, total_loss, global_step,
                    prediction_dict, train_filename, train_scale_factor, metric_ops
                ], run_metadata=run_metadata)

                print('Scaled image with {}'.format(scale_factor))
                print('Image size: {}'.format(pred_dict['image_shape']))
                draw_anchors(pred_dict)
                draw_positive_anchors(pred_dict)
                draw_top_nms_proposals(pred_dict, 0.9)
                draw_batch_proposals(pred_dict)
                draw_rpn_cls_loss(pred_dict)
                draw_rpn_bbox_pred(pred_dict)
                draw_rpn_bbox_pred_with_target(pred_dict)
                draw_rpn_bbox_pred_with_target(pred_dict, worst=False)
                draw_object_prediction(pred_dict)

                count_images += 1

                tf.logging.info('train_loss: {}'.format(train_loss))
                tf.logging.info('step: {}, filename: {}'.format(step, filename))

                if not no_log:
                    if step % save_every == 0:
                        # We don't support partial saver.
                        saver.save(sess, os.path.join(model_dir, model.scope_name), global_step=step)

                    if not debug:
                        values = sess.run(metrics)
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
