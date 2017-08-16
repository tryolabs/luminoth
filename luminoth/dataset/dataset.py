import sonnet as snt

from luminoth.utils.image import resize_image


class Dataset(snt.AbstractModule):
    """Abstract dataset module.

    This module implements some of the basic functionalities every dataset
    usually needs.

    Currently we only support object detection datasets, that means, datasets
    that have ground truth of bounding boxes.

    Attributes:
        dataset_dir (str): Base directory of the dataset.
        num_epochs (int): Number of epochs the dataset should iterate over.
        batch_size (int): Batch size the module should return.
        split (str): Split to consume the data from (usually "train", "val" or
            "test").
        image_min_size (int): Image minimum size, used for resizing images if
            needed.
        image_max_size (int): Image maximum size.
        random_shuffle (bool): To consume the dataset using random shuffle or
            to just use a regular FIFO queue.

    TODO: Abstract BoundingBoxesDataset.
    """
    def __init__(self, config, **kwargs):
        """
        Save general purpose attributes for Dataset module.

        Args:
            config: Config object with all the session properties.
        """
        super(Dataset, self).__init__(**kwargs)
        self._dataset_dir = config.dataset.dir
        self._num_epochs = config.train.num_epochs
        self._batch_size = config.train.batch_size
        self._split = config.dataset.split
        self._image_min_size = config.dataset.image_preprocessing.min_size
        self._image_max_size = config.dataset.image_preprocessing.max_size
        self._random_shuffle = config.train.random_shuffle

    def _resize_image(self, image, bboxes):
        """
        We need to resize image and bounding boxes when the biggest side
        dimension is bigger than `self._image_max_size` or when the smaller
        side is smaller than `self._image_min_size`.

        Then, using the ratio we used, we need to properly scale the bounding
        boxes.

        Args:
            image: Tensor with image of shape (H, W, 3).
            bboxes: Tensor with bounding boxes with shape (num_bboxes, 5).
                where we have (x_min, y_min, x_max, y_max, label) for each one.

        Returns:
            image: Tensor with scaled image.
            bboxes: Tensor with scaled (using the same factor as the image)
                bounding boxes with shape (num_bboxes, 5).
            scale_factor: Scale factor used to modify the image (1.0 means no
                change).
        """
        resized = resize_image(
            image, bboxes=bboxes, min_size=self._image_min_size,
            max_size=self._image_max_size
        )

        return resized['image'], resized['bboxes'], resized['scale_factor']
