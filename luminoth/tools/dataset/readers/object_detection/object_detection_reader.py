import abc

from luminoth.tools.dataset.readers import BaseReader


class ObjectDetectionReader(BaseReader):
    """Reads data suitable for object detection.

    Object detections needs:
        - images
        - gt_boxes with labels
    """
    def __init__(self):
        super(ObjectDetectionReader, self).__init__()

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

    @abc.abstractproperty
    def classes(self):
        """Returns a list of class names available in dataset.
        """

    def set_classes(self, classes):
        self._classes = classes
