import os
import sonnet as snt
import tensorflow as tf
import click

from .network import FasterRCNN
from .pretrained import VGG, ResNetV2
from .config import Config
from .dataset import TFRecordDataset

# debug
from tensorflow.python.client import timeline
from .utils.image_vis import *


PRETRAINED_MODULES = {
    'vgg': VGG,
    'vgg_16': VGG,
    'resnet': ResNetV2,
    'resnetv2': ResNetV2,
}


@click.command()
@click.option('--num-classes', default=20)
@click.option('--pretrained-net', default='vgg_16', type=click.Choice(PRETRAINED_MODULES.keys()))
@click.option('--pretrained-weights')
@click.option('--model-dir', default='models/')
@click.option('--checkpoint-file')
@click.option('--pretrained-checkpoint-file')
@click.option('--ignore-scope')
@click.option('--log-dir', default='/tmp/frcnn/')
@click.option('--save-every', default=100)
@click.option('--tf-debug', is_flag=True)
@click.option('--debug', is_flag=True)
@click.option('--run-name', default='train')
@click.option('--with-rcnn', default=True, type=bool)
@click.option('--no-log', is_flag=True)
@click.option('--display-every', default=1, type=int)
@click.option('--random-shuffle', is_flag=True)
@click.option('--save-timeline', is_flag=True)
@click.option('--summary-every', default=1, type=int)
@click.option('--full-trace', is_flag=True)
@click.option('--initial-learning-rate', default=0.0001, type=float)
@click.option('--learning-rate-decay', default=10000, type=int)
def train(num_classes, pretrained_net, pretrained_weights, model_dir,
          checkpoint_file, pretrained_checkpoint_file, ignore_scope, log_dir,
          save_every, tf_debug, debug, run_name, with_rcnn, no_log,
          display_every, random_shuffle, save_timeline, summary_every,
          full_trace, initial_learning_rate, learning_rate_decay):

    if debug or tf_debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    pretrained = PRETRAINED_MODULES[pretrained_net](trainable=False)
    model = FasterRCNN(
        Config, debug=debug, num_classes=num_classes, with_rcnn=with_rcnn,
    )
    dataset = TFRecordDataset(
        Config, num_classes=num_classes, random_shuffle=random_shuffle
    )
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
    # Bbox doesn't need a dimension for batch TODO: Necesitamos standarizarlo!
    # train_bboxes = tf.expand_dims(train_bboxes, 0)

    pretrained_dict = pretrained(train_image)
    prediction_dict = model(train_image, pretrained_dict['net'], train_bboxes)

    model_variables = snt.get_normalized_variable_map(
        model, tf.GraphKeys.GLOBAL_VARIABLES
    )
    if ignore_scope:
        total_model_variables = len(model_variables)
        model_variables = {
            k: v for k, v in model_variables.items() if ignore_scope not in k
        }
        new_total_model_variables = len(model_variables)
        tf.logging.info('Not loading/saving {} variables with scope "{}"'.format(
            total_model_variables - new_total_model_variables, ignore_scope))

        partial_saver = tf.train.Saver(var_list=model_variables)

    load_op = pretrained.load_weights(
        checkpoint_file=pretrained_weights
    )

    total_loss = model.loss(prediction_dict)

    model_saver = snt.get_saver(model, name='fasterrcnn_saver')
    pretrained_saver = snt.get_saver(pretrained, name='pretrained_saver')

    global_step = tf.get_variable(
        name="global_step", shape=[], dtype=tf.int64,
        initializer=tf.zeros_initializer(), trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP]
    )

    learning_rate = tf.train.exponential_decay(
        learning_rate=initial_learning_rate, global_step=global_step,
        decay_steps=learning_rate_decay, decay_rate=0.96, staircase=True,
        name='learning_rate_with_decay'
    )

    tf.summary.scalar('losses/learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9
    )
    trainable_vars = snt.get_variables_in_module(model)
    if Config.PRETRAINED_TRAINABLE:
        trainable_vars += snt.get_variables_in_module(pretrained)
    else:
        tf.logging.info('Not training variables from pretrained module')

    grads_and_vars = optimizer.compute_gradients(total_loss, trainable_vars)

    # Clip by norm. Grad can be null when not training some modules.
    with tf.name_scope('clip_gradients_by_norm'):
        grads_and_vars = [
            (tf.check_numerics(tf.clip_by_norm(gv[0], 10.), 'Invalid gradient'), gv[1])
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

    metric_ops = tf.get_collection('metric_ops')
    metrics = tf.get_collection('metrics')

    tf.logging.info('Starting training for {}'.format(model))

    summarizer = tf.summary.merge([
        tf.summary.merge_all(),
        model.summary,
    ])

    with tf.Session() as sess:

        if tf_debug:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(init_op)  # initialized variables
        sess.run(load_op)  # load pretrained weights

        # Restore all variables from checkpoint file
        if checkpoint_file:
            # TODO: We are WAY better than this.

            # If ignore_scope is set, we don't load those variables from checkpoint.
            if ignore_scope:
                partial_saver.restore(sess, checkpoint_file)
            else:
                model_saver.restore(sess, checkpoint_file)

        if pretrained_checkpoint_file:
            pretrained_saver.restore(sess, pretrained_checkpoint_file)

        if not no_log:
            writer = tf.summary.FileWriter(
                os.path.join(log_dir, run_name), sess.graph
            )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        count_images = 0
        step = 0

        if tf_debug:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        try:
            while not coord.should_stop():
                run_metadata = None
                if (step + 1) % summary_every == 0:
                    run_metadata = tf.RunMetadata()

                run_options = None
                if full_trace:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE
                    )

                (_, summary, train_loss, step, pred_dict, filename,
                 scale_factor, *_) = sess.run(
                    [
                        train_op, summarizer, total_loss, global_step,
                        prediction_dict, train_filename, train_scale_factor,
                        metric_ops
                    ], run_metadata=run_metadata, options=run_options)

                if not no_log and step % summary_every == 0:
                    writer.add_summary(summary, step)
                    writer.add_run_metadata(
                        run_metadata, str(step)
                    )

                if debug and step % display_every == 0:
                    print('Scaled image with {}'.format(scale_factor))
                    print('Image size: {}'.format(pred_dict['image_shape']))
                    draw_anchors(pred_dict)
                    draw_positive_anchors(pred_dict)
                    draw_top_nms_proposals(pred_dict, 0.9)
                    draw_batch_proposals(pred_dict, display_anchor=True)
                    draw_batch_proposals(pred_dict, display_anchor=False)
                    draw_rpn_cls_loss(pred_dict, foreground=True, topn=10, worst=True)
                    draw_rpn_cls_loss(pred_dict, foreground=True, topn=10, worst=False)
                    draw_rpn_cls_loss(pred_dict, foreground=False, topn=10, worst=True)
                    draw_rpn_cls_loss(pred_dict, foreground=False, topn=10, worst=False)
                    draw_rpn_bbox_pred(pred_dict)
                    draw_rpn_bbox_pred_with_target(pred_dict)
                    draw_rpn_bbox_pred_with_target(pred_dict, worst=False)
                    if with_rcnn:
                        draw_rcnn_cls_batch(pred_dict)
                        draw_rcnn_input_proposals(pred_dict)
                        draw_rcnn_cls_batch_errors(pred_dict, worst=False)
                        draw_rcnn_reg_batch_errors(pred_dict)
                        draw_object_prediction(pred_dict)

                    if save_timeline:
                        run_tmln = timeline.Timeline(
                            run_metadata.step_stats)
                        chrome_trace = run_tmln.generate_chrome_trace_format()
                        with open('timeline_{}.json'.format(step), 'w') as f:
                            f.write(chrome_trace)

                count_images += 1

                tf.logging.info('train_loss: {}'.format(train_loss))
                tf.logging.info('step: {}, file: {}'.format(step, filename))

                if not no_log:
                    if step % save_every == 0:
                        # We don't support partial saver.
                        model_saver.save(
                            sess,
                            os.path.join(model_dir, run_name, model.scope_name),
                            global_step=step
                        )
                        pretrained_saver.save(
                            sess,
                            os.path.join(model_dir, run_name, pretrained.scope_name),
                            global_step=step
                        )

        except tf.errors.OutOfRangeError:
            tf.logging.info('iter = {}, train_loss = {:.2f}'.format(
                step, train_loss))
            tf.logging.info('finished training -- epoch limit reached')
            tf.logging.info('count_images = {}'.format(count_images))
        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)


if __name__ == '__main__':
    train()
