import click
import os
import tensorflow as tf

from datetime import datetime

from .detector import inputs, inference, metrics, loss, optimizer, PRINT_EVERY


def run_training(global_step, train_op, total_loss, log_dir, model_dir):
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

    print("graph built, starting the session")
    with tf.Session() as sess:
        # Run the initializer, then the rest.
        sess.run(init_op)

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
                _, summary, train_loss, step, *_ = sess.run([
                    train_op, summarizer, total_loss, global_step, metric_ops
                ])

                # Run the metric operations to retrieve the values.
                # Don't print per-class AUC while training.
                values = sess.run(metrics)
                metrics_report = ', '.join([
                    f'{metric.op.name} = {value:.2f}'
                    for metric, value in zip(metrics, values)
                    if metric.op.name.startswith('auc')
                ])

                # Get and track metrics for validation and training sets.
                if step % PRINT_EVERY == 0:
                    writer.add_summary(summary, step)

                    line = 'iter = {}, loss = {:.2f}, {}'
                    print(line.format(step, train_loss, metrics_report))

                    saver.save(sess, os.path.join(model_dir, 'model'), step)

        except tf.errors.OutOfRangeError:
            if step % PRINT_EVERY != 0:
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
def train(data_dir, log_dir, model_dir):
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

    # Inputs for graph.
    X, y_true = inputs(data_dir)

    # Graph architecture.
    y_pred = inference(X)

    # Define metrics-related operations.
    metrics(y_pred, y_true)

    # Data and regularization loss operations.
    total_loss = loss(y_pred, y_true)

    # Add the loss summary out here so we don't add it for the eval too.
    tf.summary.scalar('loss', total_loss)

    # Training operation; automatically updates all variables using SGD.
    global_step, train_op = optimizer(total_loss)

    # Run the training loop.
    run_training(global_step, train_op, total_loss, log_dir, model_dir)
