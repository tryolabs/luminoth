import abc
import six

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

    Additionally, must use the `_per_class_counter` variable to honor the max
    number of examples per class in an efficient way.
    """
    def __init__(self, only_classes=None, only_images=None,
                 max_per_class=None, **kwargs):
        """
        Args:
            - only_classes: string or list of strings used as a class
                whitelist.
            - only_images: string or list of strings used as a image_id
                whitelist.
            - max_per_class: max number of examples to use per class.
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
        self._max_per_class = max_per_class
        self._per_class_counter = Counter()

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
        Returns all the classes available in the dataset.
        """

    def _filter_total(self, original_total_records):
        """
        Filters total number of records in dataset based on reader options
        used.
        """
        if self._only_images:  # not None and not empty
            return len(self._only_images)

        if self._max_per_class is not None:
            # We don't know the exact total, but we know the bound, so
            # return that (only to make the progressbar of the writer more
            # accurate).
            return len(self.classes) * self._max_per_class

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

    def _should_skip(self, image_id=None, label=None):
        """
        Determine if we should skip the current image, based on the options
        used for the reader.

        Args:
            - image_id: String with id of image.
            - label: String or number with the corresponding label.

        Returns:
            bool: True if record should be skipped, False if not.
        """
        if self._only_images is not None and image_id is not None:
            # Skip because the current image_id was not asked for.
            if image_id not in self._only_images:
                return True

        if self._max_per_class is not None and label is not None:
            # Skip because current class is maxed out
            if self._per_class_counter[label] >= self._max_per_class:
                return True

        return False

    def _stop_iteration(self):
        return self.yielded_records == self.total

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
