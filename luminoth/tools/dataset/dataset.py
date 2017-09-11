import abc
import click
import json
import os
import random
import tensorflow as tf


class InvalidDataDirectory(Exception):
    """
    Error raised when the chosen intput directory for the dataset is not valid.
    """


class DatasetTool(object):

    VALID_SPLITS = ['train', 'val', 'test']

    def __init__(self):
        super(DatasetTool, self).__init__()

    @abc.abstractmethod
    def is_valid(self):
        """
        Check that the dataset is valid.

        Raises:
            InvalidDataDirectory: When the instance directory is not valid.
        """

    @abc.abstractmethod
    def read_classes(self):
        """
        Read and return the total list of classes available in the dataset.

        The sorting is important since it defines the way classes are
        represented, assigning numerical 0-based labels based on the position.

        Args:
            root: Root directory of dataset.

        Returns:
            classes: List of sorted classes.
        """

    @abc.abstractmethod
    def load_split(self, split='train'):
        """
        Returns the image identifiers corresponding to the split `split`
        (values: 'train', 'val', 'test').

        Args:
            root: Root directory of dataset.
            split: Data split to return.

        Returns:
            filenames (generator):
        """

    @abc.abstractmethod
    def get_split_size(self, split):
        """
        Returns the total number of samples found in the split `split`.

        Args:
            split: Data split.

        Returns:
            total_samples: Total number of samples in split `split` in dataset.
        """

    @abc.abstractmethod
    def get_image_path(self, image_id):
        """
        Get path for image based on root directory and image id.

        Args:
            root: Root directory of dataset.
            image_id: Image id.

        Returns:
            image_path: Full path to image file.
        """

    @abc.abstractmethod
    def get_image_annotation(self, image_id):
        """
        Get path for image annotation file.

        Args:
            root: Root directory of dataset.
            image_id: Image id.

        Returns:
            image_annotation: Full path to image annotation file.
        """

    @abc.abstractmethod
    def image_to_example(self, classes, image_id):
        """
        Args:
            data_dir: Root directory of dataset.
            classes: Sorted list of classes.
            image_id: Image id.

        Returns:
            example: tf.train.SequenceExample containing all the image data.
        """


class RecordSaver():
    def __init__(self, dataset, output_dir, ignore_splits=None,
                 only_filename=None, limit_examples=None, limit_classes=None,
                 seed=None):
        self.dataset = dataset
        self.output_dir = output_dir
        self.ignore_splits = ignore_splits
        self.only_filename = only_filename
        self.limit_examples = limit_examples
        self.limit_classes = limit_classes
        random.seed(seed)

    @property
    def classes(self):
        if hasattr(self, '_classes'):
            return self._classes

        # Read all classes from dataset
        classes = self.dataset.read_classes()
        if self.limit_classes:
            # Overwrite limit to avoid confusing names.
            self.limit_classes = min(self.limit_classes, len(classes))

        if self.limit_classes and self.limit_classes < len(classes):
            # Sort classes as they are supposed to be sorted from dataset.
            classes = sorted(random.sample(classes, self.limit_classes))
            tf.logging.info('Limiting to {} classes: {}'.format(
                self.limit_classes, classes
            ))

        self._classes = classes
        return self._classes

    def get_classes_file(self):
        if self.only_filename:
            classes_filename = 'classes-{}.json'.format(self.only_filename)
        elif self.limit_examples:
            classes_filename = 'classes-top{}-{}classes.json'.format(
                self.limit_examples, self.limit_classes
            )
        else:
            classes_filename = 'classes.json'

        classes_file = os.path.join(self.output_dir, classes_filename)
        return classes_file

    def get_record_file(self, split):
        if self.only_filename:
            record_filename = '{}-{}.tfrecords'.format(
                split, self.only_filename
            )
        elif self.limit_examples:
            record_filename = '{}-top{}-{}classes.tfrecords'.format(
                split, self.limit_examples, self.limit_classes
            )
        else:
            record_filename = '{}.tfrecords'.format(split)

        return os.path.join(self.output_dir, record_filename)

    def save(self):
        tf.logging.info('Saving output_dir = {}'.format(self.output_dir))
        if not tf.gfile.Exists(self.output_dir):
            tf.gfile.MakeDirs(self.output_dir)

        # Save classes in simple json format for later use.
        classes_file = self.get_classes_file()
        json.dump(self.classes, tf.gfile.GFile(classes_file, 'w'))

        # Each dataset may contain different valid splits.
        splits = [
            s for s in self.dataset.VALID_SPLITS
            if s not in set(self.ignore_splits)
        ]
        tf.logging.debug(
            'Generating outputs for splits = {}'.format(", ".join(splits)))

        for split in splits:
            split_size = self.dataset.get_split_size(split)
            if self.limit_examples:
                split_size = min(self.limit_examples, split_size)

            tf.logging.info('Converting split = {}'.format(split))
            record_file = self.get_record_file(split)
            writer = tf.python_io.TFRecordWriter(record_file)

            total_examples = 0
            with click.progressbar(self.dataset.load_split(split),
                                   length=split_size) as split_lines:
                for image_id in split_lines:
                    if (not self.only_filename or
                       self.only_filename == image_id):
                        # Using limit on classes it's possible for an
                        # image_to_example to return None (because no classes
                        # match).
                        example = self.dataset.image_to_example(
                            self.classes, image_id
                        )
                        if example:
                            total_examples += 1
                            writer.write(example.SerializeToString())

                    stop_saving = (
                        self.limit_examples and
                        total_examples == self.limit_examples
                    )
                    if stop_saving:
                        break

            tf.logging.info(
                'Saved split {} to "{}"'.format(split, record_file)
            )
