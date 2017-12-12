import abc
import random
import six

from luminoth.tools.dataset.readers import BaseReader


class ObjectDetectionReader(BaseReader):
    """Reads data suitable for object detection.

    Object detections needs:
        - images
        - gt_boxes with labels

    For implementing a subclass of object detection one needs to implement the
    following methods:
        - __init__(data_dir, split, **kwargs)
        - get_total(self)
            Get total amount of records.
        - get_classes(self)
            Get all classes in records.
        - iterate(self)
            Iterate over all records.
    """
    def __init__(self, only_classes=None, only_images=None,
                 limit_examples=None, limit_classes=None, seed=None, **kwargs):
        """
        Args:
            - only_classes: string or list of strings used as a class
                whitelist.
            - only_images: string or list of strings used as a image_id
                whitelist.
            - limit_examples: limit number of examples to use.
            - limit_classes: limit number of classes to use.
            - seed: seed for random.
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

        self._limit_examples = limit_examples
        self._limit_classes = limit_classes
        random.seed(seed)

        self._total = None
        self._classes = None

    @property
    def total(self):
        if self._total is None:
            self._total = self._filter_total(self.get_total())
        return self._total

    @property
    def classes(self):
        if self._classes is None:
            self._classes = self._filter_classes(self.get_classes())
        return self._classes

    @abc.abstractmethod
    def get_total(self):
        """Returns the total amount of records in the dataset.
        """

    @abc.abstractmethod
    def get_classes(self):
        """Returns all the classes available in the dataset.
        """

    def _filter_total(self, original_total_records):
        """Filters total number of records in dataset based on reader options
        used.
        """
        # Define smaller number of records when limiting examples.
        if self._only_images:  # not none and not empty
            new_total = len(self._only_images)
        elif self._limit_examples is not None and self._limit_examples > 0:
            new_total = min(self._limit_examples, original_total_records)
        else:
            new_total = original_total_records

        return new_total

    def _filter_classes(self, original_classes):
        """Filters classes based on reader options used.
        """
        if self._only_classes:  # not None and not empty
            new_classes = sorted(self._only_classes)
        # Choose random classes when limiting them
        elif self._limit_classes is not None and self._limit_classes > 0:
            total_classes = min(len(original_classes), self._limit_classes)
            new_classes = sorted(
                random.sample(original_classes, total_classes)
            )
        else:
            new_classes = original_classes

        return new_classes

    def _is_valid(self, image_id):
        """
        Checks if image_id is valid based on the reader options used.

        Args:
            - image_id: String with id of image.

        Returns:
            bool
        """
        return self._only_images is None or image_id in self._only_images

    def _stop_iteration(self):
        return self.yielded_records == self.total

    @abc.abstractmethod
    def iterate(self):
        """Iterate over object detection records read from the dataset source.

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

    def set_classes(self, classes):
        self._classes = classes
