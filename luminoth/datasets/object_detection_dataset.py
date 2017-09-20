import sonnet as snt
import tensorflow as tf

from luminoth.utils.image import (
    resize_image, flip_image, random_patch, random_resize, random_distortion
)
DATA_AUGMENTATION_STRATEGIES = {
    'flip': flip_image,
    'patch': random_patch,
    'resize': random_resize,
    'distortion': random_distortion
}


class ObjectDetectionDataset(snt.AbstractModule):
    """Abstract object detector dataset module.

    This module implements some of the basic functionalities every object
    detector dataset usually needs.

    Object detection datasets, are datasets that have ground-truth information
    consisting of bounding boxes.

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
    """
    def __init__(self, config, **kwargs):
        """
        Save general purpose attributes for Dataset module.

        Args:
            config: Config object with all the session properties.
        """
        super(ObjectDetectionDataset, self).__init__(**kwargs)
        self._dataset_dir = config.dataset.dir
        self._num_epochs = config.train.num_epochs
        self._batch_size = config.train.batch_size
        self._split = config.dataset.split
        self._image_min_size = config.dataset.image_preprocessing.min_size
        self._image_max_size = config.dataset.image_preprocessing.max_size
        self._random_shuffle = config.train.random_shuffle
        # In case no keys are defined, default to empty list.
        self._data_augmentation = config.dataset.data_augmentation or []
        self._seed = config.train.seed

    def _build():
        pass

    def _augment(self, image, bboxes, default_prob=0.5):
        """Applies different data augmentation techniques.

        Uses the list of data augmentation configurations, each data
        augmentation technique has a probability assigned to it (or just uses
        the default value for the dataset).

        Procedures are applied sequentially on top of each other according to
        the order defined in the config.

        TODO: We need a better way to ensure order using YAML config without
        ending up with a list of single-key dictionaries.

        Args:
            image: A Tensor of shape (height, width, 3).
            bboxes: A Tensor of shape (total_bboxes, 5).

        Returns:
            image: A Tensor of shape (height, width, 3).
            bboxes: A Tensor of shape (total_bboxes, 5) of type tf.int32.
        """
        applied_data_augmentation = []
        for aug_config in self._data_augmentation:
            if len(aug_config.keys()) != 1:
                raise ValueError(
                    'Invalid data_augmentation definition: "{}"'.format(
                        aug_config))

            aug_type = list(aug_config.keys())[0]
            if aug_type not in DATA_AUGMENTATION_STRATEGIES:
                tf.logging.warning(
                    'Invalid data augmentation strategy "{}". Ignoring'.format(
                        aug_type))
                continue

            aug_config = aug_config[aug_type]
            aug_fn = DATA_AUGMENTATION_STRATEGIES[aug_type]

            random_number = tf.random_uniform([], seed=self._seed)
            prob = tf.to_float(aug_config.pop('prob', default_prob))
            apply_aug_strategy = tf.less(random_number, prob)

            augmented = tf.cond(
                apply_aug_strategy,
                lambda: aug_fn(image, bboxes, **aug_config),
                lambda: {'image': image, 'bboxes': bboxes}
            )
            image = augmented['image']
            bboxes = augmented.get('bboxes')

            applied_data_augmentation.append({aug_type: apply_aug_strategy})

        return image, bboxes, applied_data_augmentation

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
