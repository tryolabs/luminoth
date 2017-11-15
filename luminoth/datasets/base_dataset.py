import os
import tensorflow as tf
import sonnet as snt

from luminoth.datasets.exceptions import InvalidDataDirectory


class BaseDataset(snt.AbstractModule):
    def __init__(self, config, **kwargs):
        super(BaseDataset, self).__init__(**kwargs)
        self._dataset_dir = config.dataset.dir
        self._num_epochs = config.train.num_epochs
        self._batch_size = config.train.batch_size
        self._split = config.dataset.split
        self._random_shuffle = config.train.random_shuffle
        self._seed = config.train.seed

        self._total_queue_ops = 20

    def _build(self):
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

        values, dtypes, names = self.read_record(raw_record)

        if self._random_shuffle:
            queue = tf.RandomShuffleQueue(
                capacity=100,
                min_after_dequeue=0,
                dtypes=dtypes,
                names=names,
                name='tfrecord_random_queue',
                seed=self._seed
            )
        else:
            queue = tf.FIFOQueue(
                capacity=100,
                dtypes=dtypes,
                names=names,
                name='tfrecord_fifo_queue'
            )

        # Generate queueing ops for QueueRunner.
        enqueue_ops = [queue.enqueue(values)] * self._total_queue_ops
        self.queue_runner = tf.train.QueueRunner(queue, enqueue_ops)

        tf.train.add_queue_runner(self.queue_runner)

        return queue.dequeue()
