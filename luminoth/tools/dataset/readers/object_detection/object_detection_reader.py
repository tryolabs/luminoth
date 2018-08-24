import abc
import six
import tensorflow as tf

from collections import Counter

from luminoth.tools.dataset.readers import BaseReader


class ObjectDetectionReader(BaseReader):
    """
    Reads data suitable for object detection.

    The object detection task needs the following information:
        - images.
        - ground truth, rectangular bounding boxes with associated labels.

    For implementing a subclass of object detection, one needs to implement
    the following methods:
        - __init__
        - get_total
        - get_classes
        - iterate

    Optionally:
        - pretty_name

    Additionally, must use the `_per_class_counter` variable to honor the max
    number of examples per class in an efficient way.
    """
    def __init__(self, only_classes=None, only_images=None,
                 limit_examples=None, class_examples=None, **kwargs):
        """
        Args:
            - only_classes: string or list of strings used as a class
                whitelist.
            - only_images: string or list of strings used as a image_id
                whitelist.
            - limit_examples: max number of examples (images) to use.
            - class_examples: finish when every class has this approximate
                number of examples.
        """
        super(ObjectDetectionReader, self).__init__()
        if isinstance(only_classes, six.string_types):
            # We can get one class as string.
            only_classes = only_classes.split(',')
        self._only_classes = only_classes

        if isinstance(only_images, six.string_types):
            # We can get one class as string.
            only_images = only_images.split(',')
        self._only_images = only_images

        self._total = None
        self._classes = None

        self._limit_examples = limit_examples
        self._class_examples = class_examples
        self._per_class_counter = Counter()
        self._maxed_out_classes = set()

    @property
    def total(self):
        if self._total is None:
            self._total = self._filter_total(self.get_total())
        return self._total

    @property
    def classes(self):
        if self._classes is None:
            self._classes = self.get_classes()
        return self._classes

    @classes.setter
    def classes(self, classes):
        self._classes = classes

    @abc.abstractmethod
    def get_total(self):
        """
        Returns the total number of records in the dataset.
        """

    @abc.abstractmethod
    def get_classes(self):
        """
        Returns a list of all the classes available in the dataset.
        """

    def pretty_name(self, label):
        """
        Return the "pretty" name for easy human identification of a given
        label.
        """
        return label

    def _filter_total(self, original_total_records):
        """
        Filters total number of records we will need to iterate over in dataset
        based on reader options used.
        """
        if self._only_images:  # not None and not empty
            return len(self._only_images)

        if self._limit_examples is not None and self._limit_examples > 0:
            return min(self._limit_examples, original_total_records)

        # With _class_examples we potentially have to iterate over every
        # record, so we don't know the total ahead of time.

        return original_total_records

    def _filter_classes(self, original_classes):
        """
        Filters classes based on reader options used.
        """
        if self._only_classes:  # not None and not empty
            new_classes = sorted(self._only_classes)
        else:
            new_classes = list(original_classes) if original_classes else None

        return new_classes

    def _should_skip(self, image_id):
        """
        Determine if we should skip the current image, based on the options
        used for the reader.

        Args:
            - image_id: String with id of image.

        Returns:
            bool: True if record should be skipped, False if not.
        """
        if self._only_images is not None and image_id is not None:
            # Skip because the current image_id was not asked for.
            if image_id not in self._only_images:
                return True

        return False

    def _all_maxed_out(self):
        # Every class is maxed out
        if self._class_examples is not None:
            return len(self._maxed_out_classes) == len(self.classes)

        return False

    def _stop_iteration(self):
        if self.yielded_records == self.total:
            return True

        if self._all_maxed_out():
            return True

        return False

    def _will_add_record(self, record):
        """
        Called whenever a new record is to be added.
        """
        # Adjust per-class counter from totals from current record
        for box in record['gt_boxes']:
            self._per_class_counter[self.classes[box['label']]] += 1

        if self._class_examples is not None:
            # Check which classes we have maxed out.
            old_maxed_out = self._maxed_out_classes.copy()

            self._maxed_out_classes |= set([
                label
                for label, count in self._per_class_counter.items()
                if count >= self._class_examples
            ])

            for label in self._maxed_out_classes - old_maxed_out:
                tf.logging.debug('Maxed out "{}" at {} examples'.format(
                    self.pretty_name(label),
                    self._per_class_counter[label]
                ))

    @abc.abstractmethod
    def iterate(self):
        """
        Iterate over object detection records read from the dataset source.

        Returns:
            iterator of records of type `dict` with the following keys:
                width (int): Image width.
                height (int): Image height.
                depth (int): Number of channels in the image.
                filename (str): Filename (or image id)
                image_raw (bytes): Read image as bytes
                gt_boxes (list of dicts):
                    label (int): Label index over the possible classes.
                    xmin (int): x value for top-left point.
                    ymin (int): y value for top-left point.
                    xmax (int): x value for bottom-right point.
                    ymax (int): y value for bottom-right point.
        """
