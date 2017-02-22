import click
import tensorflow as tf
import os

from datetime import datetime

from .voc import read_classes


# TODO: Specify as argument?
IMAGE_SIZE = 200


def inputs(data_dir, num_epochs, batch_size):
    classes = read_classes(data_dir)

    train_path = os.path.join(data_dir, 'tf', 'train.tfrecords')
    # val_path = os.path.join(data_dir, 'tf', 'val.tfrecords')
    # test_path = os.path.join(data_dir, 'tf', 'test.tfrecords')

    filename_queue = tf.train.string_input_producer(
        [train_path], num_epochs=num_epochs,
    )

    # TODO: Can I add multiple readers if all the samples are in a single file?
    # TODO: Maybe it's not even needed, should check fill levels for queues.
    reader = tf.TFRecordReader()
    _, raw_record = reader.read(filename_queue)

    # TODO: Could move to `voc.py` so both writing and reading code is on the
    # same place.
    # TODO: Why FixedLenFeature for variable-length images? (i.e. JPEGs have
    # variable sizes).
    # TODO: Why do I not need to add shape to the image and but do for the
    # label?
    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([len(classes)], tf.int64),
    }
    example = tf.parse_single_example(raw_record, features)

    # TODO: The fact that it's a JPEG file should also be in `voc.py`.
    # TODO: Images are around ~500 pixels, should resize first when decoding?
    # Decode and preprocess the example (crop, adjust mean and variance).
    # image_jpeg = tf.decode_raw(example['image_raw'], tf.string)
    image_raw = tf.image.decode_jpeg(example['image_raw'])
    resized_image = tf.image.resize_image_with_crop_or_pad(
        image_raw, IMAGE_SIZE, IMAGE_SIZE
    )
    image = tf.image.per_image_standardization(resized_image)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    label = tf.cast(example['label'], tf.float32)

    # Batch the samples.
    min_after_dequeue = 200  # Dataset isn't very big.
    # TODO: What *is* this num_threads? Threads training or threads building
    # batches? "[...] a third set of threads dequeues these input records to
    # construct batches and runs them through training operations."
    num_threads = 1
    capacity = min_after_dequeue + (num_threads + 0.2) * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    # TODO: With this, are there two or three queues? Is there a queue for the
    # batches or does it end on single examples?

    return image_batch, label_batch


def inference(X, num_classes, reg):
    """
    Build the model for performing inference on input X.

    Adds regularization losses to the TF `losses` collection.
    """
    hidden_size = 500
    conv_size = [3, 3, 3, 32]

    Wconv = tf.Variable(tf.random_normal(conv_size, stddev=0.01))
    tf.add_to_collection('losses', reg * tf.nn.l2_loss(Wconv))

    bconv = tf.Variable(tf.zeros([conv_size[-1]]))
    conv = tf.nn.relu(tf.nn.conv2d(
        X, Wconv,
        strides=[1, 1, 1, 1],
        padding='SAME'
    ) + bconv)

    pool = tf.nn.max_pool(
        conv,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )

    # TODO: Better shape handling.
    pool_shape = pool.get_shape()[1:]
    hidden_shape = [
        int(pool_shape[0] * pool_shape[1] * pool_shape[2]),
        hidden_size
    ]

    W1 = tf.Variable(tf.random_normal(hidden_shape, stddev=0.01))
    tf.add_to_collection('losses', reg * tf.nn.l2_loss(W1))

    b1 = tf.Variable(tf.zeros([hidden_size]))

    hidden = tf.nn.relu(tf.matmul(
        tf.reshape(pool, [-1, hidden_shape[0]]), W1
    ) + b1)

    W2 = tf.Variable(
        tf.random_normal([hidden_size, num_classes], stddev=0.01)
    )
    tf.add_to_collection('losses', reg * tf.nn.l2_loss(W2))

    b2 = tf.Variable(tf.zeros([num_classes]))

    y_pred = tf.matmul(hidden, W2) + b2

    return y_pred


def metrics(logits, labels):
    # TODO: Doesn't make much sense over training data, averaging with initial
    # (bad) predictions.
    auc, update_auc_op = tf.metrics.auc(labels, logits, curve='PR')
    tf.add_to_collection('metric_ops', update_auc_op)
    tf.add_to_collection('metrics', auc)
    tf.summary.scalar('auc', auc)


def loss(logits, labels):
    data_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    tf.add_to_collection('losses', data_loss)

    total_loss = tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('total_loss', total_loss)

    return total_loss


def optimizer(total_loss, learning_rate, beta1, beta2):
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2
    ).minimize(total_loss, global_step=global_step)

    return global_step, train_op


def train(global_step, train_op, total_loss, log_dir, model_dir, print_every):
    # Merge all summary values.
    summarizer = tf.summary.merge_all()

    # Operation to save and restore variables on the graph.
    saver = tf.train.Saver()

    # Create initializer for variables. Queue-related variables need a special
    # initializer.
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    metric_ops = tf.get_collection('metric_ops')
    metrics = tf.get_collection('metrics')

    print("graph built, starting the session")
    with tf.Session() as sess:
        # Run the initializer, then the rest.
        sess.run(init_op)

        # Create seaprate summary writers for training and validation data.
        train_writer = tf.summary.FileWriter(f'{log_dir}/train', sess.graph)

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

                # Run the metric operations.
                values = sess.run(metrics)
                metrics_report = ', '.join([
                    f'{metric.op.name} = {value:.2f}'
                    for metric, value in zip(metrics, values)
                ])

                # Get and track metrics for validation and training sets.
                if step % print_every == 0:
                    train_writer.add_summary(summary, step)

                    line = 'iter = {}, loss = {:.2f}, {}'
                    print(line.format(step, train_loss, metrics_report))

                    saver.save(sess, os.path.join(log_dir, 'conv'), step)

        except tf.errors.OutOfRangeError:
            if step % print_every != 0:
                line = 'iter = {}, train_loss = {:.2f}, {}'
                print(line.format(step, train_loss), metrics_report)
            print('finished training -- epoch limit reached')
        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)

        # Saves the final variables of the graph to `MODEL_DIR`.
        # TODO: Save the best overall every epoch/batch, not the last one.
        save_path = saver.save(sess, model_dir)
        print()
        print(f'saving result to save_path = {save_path}')


@click.command()
@click.option('--data-dir', default='datasets/voc/')
@click.option('--log-dir', default='logs/')
@click.option('--model-dir', default='models/')
@click.option('--num-epochs', default=5)
@click.option('--batch-size', default=32)
@click.option('--reg', default=1e-4)
@click.option('--learning_rate', default=0.001)
@click.option('--beta1', default=0.9)
@click.option('--beta2', default=0.999)
@click.option('--print-every', default=5)
def main(data_dir, log_dir, model_dir, num_epochs, batch_size, reg,
         learning_rate, beta1, beta2, print_every):

    # Save each session log using the date they run.
    log_dir = os.path.join(
        log_dir, str(datetime.now()).split('.')[0].replace(' ', '_')
    )

    # Print selected options for sanity check.
    print(f"data_dir = {data_dir}")
    print(f"log_dir = {log_dir}")
    print(f"model_dir = {model_dir}")

    print(f"num_epochs = {num_epochs}")
    print(f"batch_size = {batch_size}")

    print(f"reg = {reg}")
    print(f"learning_rate = {learning_rate}")
    print(f"beta1 = {beta1}")
    print(f"beta2 = {beta2}")

    print(f"print_every = {print_every}")

    # Inputs for graph.
    X, y_true = inputs(data_dir, num_epochs, batch_size)

    # Inputs and outputs dimensions.
    num_classes = int(y_true.shape[1])

    # Graph architecture.
    y_pred = inference(X, num_classes, reg)

    # Metrics operations.
    metrics(y_pred, y_true)

    # Data and regularization loss operations.
    total_loss = loss(y_pred, y_true)

    # Training operation; automatically updates all variables using SGD.
    global_step, train_op = optimizer(total_loss, learning_rate, beta1, beta2)

    # Perform the training.
    train(
        global_step, train_op, total_loss, log_dir, model_dir, print_every
    )
