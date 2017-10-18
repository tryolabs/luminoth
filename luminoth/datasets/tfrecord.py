import os
import tensorflow as tf

from luminoth.tools.dataset.dataset import InvalidDataDirectory
from .object_detection_dataset import ObjectDetectionDataset


class TFRecordDataset(ObjectDetectionDataset):
    """
    Attributes:
        context_features (dict): Context features used to parse fixed sized
            tfrecords.
        sequence_features (dict): Sequence features used to parse the variable
            sized part of tfrecords (for ground truth bounding boxes).

    """
    def __init__(self, config, name='tfrecord_dataset', **kwargs):
        """
        Args:
            config: Configuration file used in session.
        """
        super(TFRecordDataset, self).__init__(config, name=name, **kwargs)

        self._context_features = {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'filename': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
        }

        self._sequence_features = {
            'label': tf.VarLenFeature(tf.int64),
            'xmin': tf.VarLenFeature(tf.int64),
            'xmax': tf.VarLenFeature(tf.int64),
            'ymin': tf.VarLenFeature(tf.int64),
            'ymax': tf.VarLenFeature(tf.int64),
        }

    def _build(self):
        """Returns a tuple containing image, image metadata and label.

        Does not receive any input since it doesn't depend on anything inside
        the graph and it's the starting point of it.

        Returns:
            dequeue_dict ({}): Dequeued dict returning and image, bounding
                boxes, filename and the scaling factor used.

        TODO: Join filename, scaling_factor (and possible other fields) into a
        metadata.
        """
        # Find split file from which we are going to read.
        split_path = os.path.join(
            self._dataset_dir, '{}.tfrecords'.format(self._split)
        )
        if not tf.gfile.Exists(split_path):
            raise InvalidDataDirectory(
                '"{}" does not exist.'.format(split_path)
            )
        # String input producer allows for a variable number of files to read
        # from. We just know we have a single file.
        filename_queue = tf.train.string_input_producer(
            [split_path], num_epochs=self._num_epochs, seed=self._seed
        )

        # Define reader to parse records.
        reader = tf.TFRecordReader()
        _, raw_record = reader.read(filename_queue)

        # We parse variable length features (bboxes in a image) as sequence
        # features
        context_example, sequence_example = tf.parse_single_sequence_example(
            raw_record,
            context_features=self._context_features,
            sequence_features=self._sequence_features
        )

        # Decode and preprocess the example (crop, adjust mean and variance).
        # image_jpeg = tf.decode_raw(example['image_raw'], tf.string)
        image_raw = tf.image.decode_jpeg(context_example['image_raw'])
        # tf.summary.image('image_raw', image_raw, max_outputs=20)

        # Do we need per_image_standardization? Do it depend on pretrained?
        image = tf.cast(image_raw, tf.float32)
        height = tf.cast(context_example['height'], tf.int32)
        width = tf.cast(context_example['width'], tf.int32)
        image_shape = tf.stack([height, width, 3])
        image = tf.reshape(image, image_shape)

        label = self.sparse_to_tensor(sequence_example['label'])
        xmin = self.sparse_to_tensor(sequence_example['xmin'])
        xmax = self.sparse_to_tensor(sequence_example['xmax'])
        ymin = self.sparse_to_tensor(sequence_example['ymin'])
        ymax = self.sparse_to_tensor(sequence_example['ymax'])

        # Stack parsed tensors to define bounding boxes of shape (num_boxes, 5)
        bboxes = tf.stack([xmin, ymin, xmax, ymax, label], axis=1)

        # Resize images (if needed)
        image, bboxes, scale_factor = self._resize_image(image, bboxes)

        image, bboxes, applied_augmentations = self._augment(image, bboxes)

        filename = tf.cast(context_example['filename'], tf.string)

        # TODO: Send additional metadata through the queue (scale_factor,
        # applied_augmentations)

        queue_dtypes = [tf.float32, tf.int32, tf.string]
        queue_names = ['image', 'bboxes', 'filename']

        if self._random_shuffle:
            queue = tf.RandomShuffleQueue(
                capacity=100,
                min_after_dequeue=0,
                dtypes=queue_dtypes,
                names=queue_names,
                name='tfrecord_random_queue',
                seed=self._seed
            )
        else:
            queue = tf.FIFOQueue(
                capacity=100,
                dtypes=queue_dtypes,
                names=queue_names,
                name='tfrecord_fifo_queue'
            )

        # Generate queueing ops for QueueRunner.
        enqueue_ops = [queue.enqueue({
            'image': image,
            'bboxes': bboxes,
            'filename': filename,
        })] * 20

        tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

        return queue.dequeue()

    def sparse_to_tensor(self, sparse_tensor, dtype=tf.int32, axis=[1]):
        return tf.squeeze(
            tf.cast(tf.sparse_tensor_to_dense(sparse_tensor), dtype), axis=axis
        )
