import sonnet as snt
import tensorflow as tf
import numpy as np

class PascalDataset(snt.AbstractModule):

    NUM_CLASSES = 20

    def __init__(self, dataset_dir='datasets/voc', num_steps=1, batch_size=1,
                 subset='train', name='pascal_dataset'):
        super(PascalDataset, self).__init__(name=name)
        self._dataset_dir = dataset_dir
        self._num_steps = num_steps
        self._batch_size = batch_size
        self._subset = subset
        self._num_epochs = 1

        self._context_features = {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([self.NUM_CLASSES], tf.int64),
            'filename': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
        }

        self._sequence_features = {
            'name': tf.VarLenFeature(tf.string),
            'xmin': tf.VarLenFeature(tf.int64),
            'xmax': tf.VarLenFeature(tf.int64),
            'ymin': tf.VarLenFeature(tf.int64),
            'ymax': tf.VarLenFeature(tf.int64),
        }

    def _build(self):
        """Returns a tuple containing image, image metadata and label."""
        q = tf.FIFOQueue(
            self._queue_capacity, [self._dtype, self._dtype],
            shapes=[[self._num_steps, self._batch_size, self._vocab_size]]*2)

        obs, target = tf.py_func(self._get_batch, [], [tf.int32, tf.int32])
        obs = self._one_hot(obs)
        target = self._one_hot(target)
        enqueue_op = q.enqueue([obs, target])
        obs, target = q.dequeue()
        tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op]))
        return SequenceDataOpsNoMask(obs, target)


        split_path = os.path.join(
            self._dataset_dir, 'tf', f'{self._subset}.tfrecords'
        )

        filename_queue = tf.train.string_input_producer(
            [split_path], num_epochs=self._num_epochs
        )

        reader = tf.TFRecordReader()
        _, raw_record = reader.read(filename_queue)

        # We parse variable length features (bboxes in a image) as sequence features
        context_example, sequence_example = tf.parse_single_sequence_example(
            raw_record, context_features=context_features, sequence_features=sequence_features
        )