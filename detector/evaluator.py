import click
import tensorflow as tf
import time
import os

from datetime import datetime

from .detector import inputs, inference, metrics, loss


def evaluate_once(saver, writer, summarizer, model_dir, split='val'):
    # Create initializer for variables. Queue-related variables need a special
    # initializer.
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    metric_ops = tf.get_collection('metric_ops')
    metrics = tf.get_collection('metrics')

    with tf.Session() as sess:
        # Run the initializer, then the rest.
        sess.run(init_op)

        # Restore the checkpoint.
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('no checkpoint available')
            return

        # Start queue runner threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run the training operations.
                summary, *_ = sess.run([summarizer, metric_ops])

                # Run the metric operations.
                # Don't print per-class AUC on logs.
                values = sess.run(metrics)
                metrics_report = ', '.join([
                    f'{metric.op.name} = {value:.2f}'
                    for metric, value in zip(metrics, values)
                    if metric.op.name.startswith('auc')
                ])

        except tf.errors.OutOfRangeError:
            line = 'iter = {}, {}'
            writer.add_summary(summary, step)
            print(line.format(step, metrics_report))
        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)


@click.command()
@click.option('--data-dir', default='datasets/voc/')
@click.option('--log-dir', default='logs/')
@click.option('--model-dir', default='models/')
@click.option('--split', default='val')
@click.option('--interval', default=30)
def evaluate(data_dir, log_dir, model_dir, split, interval):
    """
    Run evaluation on saved model.
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
    X, y_true = inputs(data_dir, num_epochs=1, split=split)

    # Graph architecture.
    y_pred = inference(X)

    # Metrics operations.
    metrics(y_pred, y_true)

    # Data and regularization loss operations.
    batch_loss = loss(y_pred, y_true)
    total_loss, update_loss_op = tf.metrics.mean(
        batch_loss, name='loss',
        metrics_collections='metrics',
        updates_collections='metric_ops',
    )
    tf.summary.scalar('loss', total_loss)

    # Merge all summary values and create summary writer.
    summarizer = tf.summary.merge_all()
    writer = tf.summary.FileWriter(f'{log_dir}/{split}')

    # Operation to save and restore variables on the graph.
    saver = tf.train.Saver()

    while True:
        evaluate_once(saver, writer, summarizer, model_dir)
        time.sleep(interval)
