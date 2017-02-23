import tensorflow as tf
import os


IMAGE_SIZE = 200

NUM_CLASSES = 20

# TODO: Not really model-related.
NUM_EPOCHS = 5
BATCH_SIZE = 32
PRINT_EVERY = 5

L2_REG = 1e-4
BETA1 = 0.9
BETA2 = 0.999
LEARNING_RATE = 0.001


def inputs(data_dir, num_epochs=NUM_EPOCHS, split='train'):
    split_path = os.path.join(data_dir, 'tf', f'{split}.tfrecords')

    filename_queue = tf.train.string_input_producer(
        [split_path], num_epochs=num_epochs,
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
        'label': tf.FixedLenFeature([NUM_CLASSES], tf.int64),
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
    capacity = min_after_dequeue + (num_threads + 0.2) * BATCH_SIZE
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=BATCH_SIZE,
        num_threads=num_threads,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    # TODO: With this, are there two or three queues? Is there a queue for the
    # batches or does it end on single examples?

    return image_batch, label_batch


def inference(X):
    """
    Build the model for performing inference on input X.

    Adds regularization losses to the TF `losses` collection.
    """
    hidden_size = 500
    conv_size = [3, 3, 3, 32]

    Wconv = tf.Variable(tf.random_normal(conv_size, stddev=0.01))
    tf.add_to_collection('losses', L2_REG * tf.nn.l2_loss(Wconv))

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
    tf.add_to_collection('losses', L2_REG * tf.nn.l2_loss(W1))

    b1 = tf.Variable(tf.zeros([hidden_size]))

    hidden = tf.nn.relu(tf.matmul(
        tf.reshape(pool, [-1, hidden_shape[0]]), W1
    ) + b1)

    W2 = tf.Variable(
        tf.random_normal([hidden_size, NUM_CLASSES], stddev=0.01)
    )
    tf.add_to_collection('losses', L2_REG * tf.nn.l2_loss(W2))

    b2 = tf.Variable(tf.zeros([NUM_CLASSES]))

    y_pred = tf.matmul(hidden, W2) + b2

    return y_pred


def metrics(logits, labels):
    # TODO: AUC doesn't make much sense over training data, averaging with
    # initial (bad) predictions.
    normalized_logits = tf.sigmoid(logits)

    # Add one AUC metric per class, so we can see individual performance too.
    for cls in range(logits.shape[1]):
        auc, _ = tf.metrics.auc(
            labels[:, cls], normalized_logits[:, cls],
            curve='PR', name=f'iauc/{cls}',
            metrics_collections='metrics',
            updates_collections='metric_ops',
        )
        tf.summary.scalar(f'iauc/{cls}', auc)

    auc, _ = tf.metrics.auc(
        labels, normalized_logits,
        curve='PR', name='auc',
        metrics_collections='metrics',
        updates_collections='metric_ops',
    )
    tf.summary.scalar('auc', auc)


def loss(logits, labels):
    data_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    tf.add_to_collection('losses', data_loss)

    total_loss = tf.add_n(tf.get_collection('losses'))

    return total_loss


def optimizer(total_loss):
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2
    ).minimize(total_loss, global_step=global_step)

    return global_step, train_op
