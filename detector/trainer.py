import click

import os
import tensorflow as tf
slim = tf.contrib.slim

from datetime import datetime

from .detector import (
    inputs, inference, metrics, loss, optimizer, get_trainable_variables,
    NUM_EPOCHS, PRINT_EVERY
)
from .nets.inception_v3 import inception_arg_scope


def get_checkpoint_fn(checkpoint_file, checkpoint_excluded_scopes):
    if not checkpoint_file:
        return lambda x: x

    # TODO: We assume the checkpoint is InceptionV3
    print(f'loading variables from checkpoint {checkpoint_file}')
    # Get list of InceptionV3 variables defined.
    # TODO: Allow configuring type of variables
    variables = tf.contrib.framework.get_variables('InceptionV3')
    # Filter variables not needed (or with dimension problems)
    variables = tf.contrib.framework.filter_variables(variables,
        exclude_patterns=checkpoint_excluded_scopes
    )
    # We only want to load those variables from pre-trained model checkpoint
    assign_from_checkpoint = tf.contrib.framework.assign_from_checkpoint_fn(
        checkpoint_file, variables, ignore_missing_vars=True
    )

    return assign_from_checkpoint


def run_training(global_step, train_op, total_loss, log_dir, model_dir,
                 checkpoint_file, checkpoint_excluded_scopes):
    # Merge all summary values.
    summarizer = tf.summary.merge_all()

    # Operation to save variables on the graph.
    saver = tf.train.Saver()

    # Create initializer for variables. Queue-related variables need a special
    # initializer.
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    # Retrieve metric-related operations previously defined.
    metric_ops = tf.get_collection('metric_ops')
    metrics = tf.get_collection('metrics')

    assign_from_checkpoint = get_checkpoint_fn(checkpoint_file, checkpoint_excluded_scopes)

    print("graph built, starting the session")
    with tf.Session() as sess:

        # Run the initializer, then the rest.
        sess.run(init_op)

        # Assign variables to session
        # TODO: Is it ok to assign variables after initializing them?
        assign_from_checkpoint(sess)

        # Create the summary writer for the training stats.
        writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'train'), sess.graph
        )

        # Start queue runner threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print("setup complete, start training")
        try:
            while not coord.should_stop():
                # Run the training operations.
                run_metadata = tf.RunMetadata()
                _, summary, train_loss, step, *_ = sess.run([
                    train_op, summarizer, total_loss, global_step, metric_ops
                ], run_metadata=run_metadata)

                # Run the metric operations to retrieve the values.
                # Don't print per-class AUC while training.
                values = sess.run(metrics)
                metrics_report = ', '.join([
                    f'{metric.op.name} = {value:.2f}'
                    for metric, value in zip(metrics, values)
                    if metric.op.name.startswith('auc')
                ])

                writer.add_summary(summary, step)
                writer.add_run_metadata(run_metadata, f'step{step}')

                # Get and track metrics for validation and training sets.
                if step % PRINT_EVERY == 0:
                    line = 'iter = {}, loss = {:.2f}, {}'
                    print(line.format(step, train_loss, metrics_report))

                    saver.save(sess, os.path.join(model_dir, 'model'), step)

        except tf.errors.OutOfRangeError:
            line = 'iter = {}, train_loss = {:.2f}, {}'
            print(line.format(step, train_loss, metrics_report))
            print('finished training -- epoch limit reached')
        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)

        # Saves the final variables of the graph to `MODEL_DIR`.
        # TODO: Save the best overall every epoch/batch, not the last one.
        save_path = saver.save(sess, os.path.join(model_dir, 'model'), step)
        print(f'saving result to save_path = {save_path}')


@click.command()
@click.option('--data-dir', default='datasets/voc/')
@click.option('--log-dir', default='logs/')
@click.option('--model-dir', default='models/')
@click.option('--epochs', default=NUM_EPOCHS)
@click.option('--checkpoint-file')
@click.option('--trainable-scopes', multiple=True)
@click.option('--checkpoint-excluded-scopes', multiple=True)
def train(data_dir, log_dir, model_dir, epochs, checkpoint_file, trainable_scopes,
          checkpoint_excluded_scopes):
    """
    Train model.
    """

    # Save each session log using the date they run.
    log_dir = os.path.join(
        log_dir, str(datetime.now()).split('.')[0].replace(' ', '_')
    )

    # Print selected options for sanity check.
    print(f"data_dir = {data_dir}")
    print(f"log_dir = {log_dir}")
    print(f"model_dir = {model_dir}")

    tf.logging.set_verbosity(tf.logging.INFO)
    # Inputs for graph.
    # Receives epochs as parameter because the input_producer is
    # the one controlling the amount of data in X.
    X, y_true = inputs(data_dir, epochs)

    # Graph architecture.
    y_pred, _ = inference(X)

    # Define metrics-related operations.
    metrics(y_pred, y_true)

    # Data and regularization loss operations.
    total_loss = loss(y_pred, y_true)

    # Add the loss summary out here so we don't add it for the eval too.
    tf.summary.scalar('loss', total_loss)

    # List of variables which we want to train
    # (in case of fine-tuning of other operations).
    trainable_variables = get_trainable_variables(trainable_scopes)

    # Training operation; automatically updates all variables using SGD.
    global_step, train_op = optimizer(total_loss, trainable_variables)

    # Run the training loop.
    run_training(
        global_step, train_op, total_loss, log_dir, model_dir,
        checkpoint_file, checkpoint_excluded_scopes
    )
