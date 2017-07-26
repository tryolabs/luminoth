import os
import sonnet as snt
import tensorflow as tf
import click

from .network import FasterRCNN
from .pretrained import VGG, ResNetV2
from .config import Config
from .dataset import TFRecordDataset

PRETRAINED_MODULES = {
    'vgg': VGG,
    'vgg_16': VGG,
    'resnet': ResNetV2,
    'resnetv2': ResNetV2,
}

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'momentum': tf.train.MomentumOptimizer,
}

LEARNING_RATE_DECAY_METHODS = set([
    'piecewise_constant', 'exponential_decay', 'none'
])


@click.command()
@click.option('--num-classes', default=20, help='Number of classes of the dataset you are training on (only used when training with RCNN).')
@click.option('--pretrained-net', default='vgg_16', type=click.Choice(PRETRAINED_MODULES.keys()), help='Architecture for the pretrained network.')
@click.option('--pretrained-weights', help='Checkpoint file with the weights of the pretrained network.')
@click.option('--model-dir', default='models/', help='Directory to save the partial trained models.')
@click.option('--checkpoint-file', help='File for the weights of RPN and RCNN for resuming training.')
@click.option('--pretrained-checkpoint-file', help='File for the weights of the pretrained network for resuming training.')
@click.option('--ignore-scope', help='Used to ignore variables when loading from checkpoint (set to "frcnn" when loading RPN and wanting to train complete network)')
@click.option('--log-dir', default='/tmp/frcnn/', help='Directory for Tensorboard logs.')
@click.option('--save-every', default=1000, help='Save checkpoint after that many batches.')
@click.option('--tf-debug', is_flag=True, help='Create debugging Tensorflow session with tfdb.')
@click.option('--debug', is_flag=True, help='Debug mode (DEBUG log level and intermediate variables are returned)')
@click.option('--run-name', default='train', help='Run name used to log in Tensorboard and isolate checkpoints.')
@click.option('--with-rcnn', default=True, type=bool, help='Train RCNN classifier (not only RPN)')
@click.option('--no-log', is_flag=True, help='Don\'t save summary logs.')
@click.option('--display-every', default=1, type=int, help='Show image debug information every N batches (debug mode must be activated)')
@click.option('--random-shuffle', is_flag=True, help='Ingest data from dataset in random order.')
@click.option('--save-timeline', is_flag=True, help='Save timeline of execution (debug mode must be activated).')
@click.option('--summary-every', default=1, type=int, help='Save summary logs every N batches.')
@click.option('--full-trace', is_flag=True, help='Run graph session with FULL_TRACE config (for memory and running time debugging)')
@click.option('--initial-learning-rate', default=0.0001, type=float, help='Initial learning date.')
@click.option('--learning-rate-decay', default=10000, type=int, help='Decay learning date after N batches.')
@click.option('--learning-rate-decay-method', default='piecewise_constant', type=click.Choice(LEARNING_RATE_DECAY_METHODS), help='Tipo of learning rate decay to use.')
@click.option('optimizer_type', '--optimizer', default='momentum', type=click.Choice(OPTIMIZERS.keys()), help='Optimizer to use.')
@click.option('--momentum', default=0.9, type=float, help='Momentum to use when using the MomentumOptimizer.')
def train(num_classes, pretrained_net, pretrained_weights, model_dir,
          checkpoint_file, pretrained_checkpoint_file, ignore_scope, log_dir,
          save_every, tf_debug, debug, run_name, with_rcnn, no_log,
          display_every, random_shuffle, save_timeline, summary_every,
          full_trace, initial_learning_rate, learning_rate_decay,
          learning_rate_decay_method, optimizer_type, momentum):

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

    if not learning_rate_decay_method or learning_rate_decay_method == 'none':
        learning_rate = initial_learning_rate
    elif learning_rate_decay_method == 'piecewise_constant':
        learning_rate = tf.train.piecewise_constant(
            global_step, boundaries=[tf.cast(learning_rate_decay, tf.int64), ],
            values=[initial_learning_rate, initial_learning_rate * 0.1],
            name='learning_rate_piecewise_constant'
        )
    elif learning_rate_decay == 'exponential_decay':
        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate, global_step=global_step,
            decay_steps=learning_rate_decay, decay_rate=0.96, staircase=True,
            name='learning_rate_with_decay'
        )
    else:
        raise ValueError(
            'Invalid learning_rate method "{}"'.format(
                learning_rate_decay_method))

    tf.summary.scalar('losses/learning_rate', learning_rate)

    optimizer_cls = OPTIMIZERS[optimizer_type]
    if optimizer_type == 'momentum':
        optimizer = optimizer_cls(learning_rate, momentum)
    else:
        optimizer = optimizer_cls(learning_rate)

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
                    from .utils.image_vis import (
                        draw_anchors, draw_positive_anchors,
                        draw_top_nms_proposals, draw_batch_proposals,
                        draw_rpn_cls_loss, draw_rpn_bbox_pred,
                        draw_rpn_bbox_pred_with_target, draw_rcnn_cls_batch,
                        draw_rcnn_input_proposals, draw_rcnn_cls_batch_errors,
                        draw_rcnn_reg_batch_errors, draw_object_prediction
                    )
                    print('Scaled image with {}'.format(scale_factor))
                    print('Image size: {}'.format(pred_dict['image_shape']))
                    draw_anchors(pred_dict)
                    draw_positive_anchors(pred_dict)
                    draw_top_nms_proposals(pred_dict, 0.9)
                    draw_top_nms_proposals(pred_dict, 0)
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
                        from tensorflow.python.client import timeline
                        run_tmln = timeline.Timeline(
                            run_metadata.step_stats)
                        chrome_trace = run_tmln.generate_chrome_trace_format()
                        with open('timeline_{}.json'.format(step), 'w') as f:
                            f.write(chrome_trace)

                count_images += 1

                tf.logging.info('train_loss: {}'.format(train_loss))
                tf.logging.info('step: {}, file: {}'.format(step, filename))

                import ipdb; ipdb.set_trace()

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
