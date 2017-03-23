import tensorflow as tf
import os

from .nets.nets_factory import get_network_fn

IMAGE_SIZE = 299

NUM_CLASSES = 20

# TODO: Not really model-related.
NUM_EPOCHS = 30
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
        'filename': tf.FixedLenFeature([], tf.string),
    }
    example = tf.parse_single_example(raw_record, features)

    # TODO: The fact that it's a JPEG file should also be in `voc.py`.
    # TODO: Images are around ~500 pixels, should resize first when decoding?
    # Decode and preprocess the example (crop, adjust mean and variance).
    # image_jpeg = tf.decode_raw(example['image_raw'], tf.string)
    image_raw = tf.image.decode_jpeg(example['image_raw'])
    # TODO: Why use crop instead of normal resize?
    resized_image = tf.image.resize_images(
        image_raw, [IMAGE_SIZE, IMAGE_SIZE]
    )
    # resized_image = tf.image.resize_image_with_crop_or_pad(
    #     image_raw, IMAGE_SIZE, IMAGE_SIZE
    # )
    summary_image = tf.reshape(resized_image, [1, IMAGE_SIZE, IMAGE_SIZE, 3])
    tf.summary.image('resized_image', summary_image, max_outputs=20)
    image = tf.image.per_image_standardization(resized_image)
    # TODO: Why do we have to manually set_shape after resize?
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


def inference(X, is_training=True):
    """
    Build the model for performing inference on input X.
    """
    inception_v3 = get_network_fn('inception_v3', NUM_CLASSES, is_training=is_training)
    logits, end_points = inception_v3(X)
    # TODO: Remove summary from this function
    for end_point, var in end_points.items():
        tf.summary.histogram('activations/' + end_point, var)
        tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(var))

    return logits, end_points


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


def optimizer(total_loss, variables=None):
    """
    Optimize total_loss operation only training on the specified `variables`.

    TODO: Allow difference optimizers.
    TODO: Hyperparameter configuration.
    """
    global_step = tf.Variable(0, trainable=False)
    train_optimizer = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2
    )
    if variables:
        print(f'only training variables {", ".join([v.op.name for v in variables])}')
    train_op = train_optimizer.minimize(
        total_loss, global_step=global_step, var_list=variables
    )

    return global_step, train_op


def get_trainable_variables(trainable_scopes):
    if not trainable_scopes:
        # If there is not `trainable_scopes` defined then we train everything.
        return tf.trainable_variables()

    variables_to_train = []
    for scope in trainable_scopes:
        # Get trainable variables with scope `scope` from graph.
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

