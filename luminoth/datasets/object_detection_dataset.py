import tensorflow as tf

from luminoth.datasets.base_dataset import BaseDataset
from luminoth.utils.image import (
    resize_image, flip_image, random_patch, random_resize, random_distortion
)

DATA_AUGMENTATION_STRATEGIES = {
    'flip': flip_image,
    'patch': random_patch,
    'resize': random_resize,
    'distortion': random_distortion
}


class ObjectDetectionDataset(BaseDataset):
    """Abstract object detector dataset module.

    This module implements some of the basic functionalities every object
    detector dataset usually needs.

    Object detection datasets are datasets that have ground-truth information
    consisting of rectangular bounding boxes.

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

    CONTEXT_FEATURES = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'filename': tf.FixedLenFeature([], tf.string),
        'width': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
    }

    SEQUENCE_FEATURES = {
        'label': tf.VarLenFeature(tf.int64),
        'xmin': tf.VarLenFeature(tf.int64),
        'xmax': tf.VarLenFeature(tf.int64),
        'ymin': tf.VarLenFeature(tf.int64),
        'ymax': tf.VarLenFeature(tf.int64),
    }

    def __init__(self, config, name='object_detection_dataset', **kwargs):
        """
        Save general purpose attributes for Dataset module.

        Args:
            config: Config object with all the session properties.
        """
        super(ObjectDetectionDataset, self).__init__(config, **kwargs)
        self._image_min_size = config.dataset.image_preprocessing.min_size
        self._image_max_size = config.dataset.image_preprocessing.max_size
        # In case no keys are defined, default to empty list.
        self._data_augmentation = config.dataset.data_augmentation or []

    def preprocess(self, image, bboxes=None):
        """Apply transformations to image and bboxes (if available).

        Transformations are applied according to the config values.
        """
        # Resize images (if needed)
        image, bboxes, scale_factor = self._resize_image(image, bboxes)
        image, bboxes, applied_augmentations = self._augment(image, bboxes)

        return image, bboxes, {
            'scale_factor': scale_factor,
            'applied_augmentations': applied_augmentations,
        }

    def read_record(self, record):
        """Parse record TFRecord into a set a set of values, names and types
        that can be queued and then read.

        Returns:
            - queue_values: Dict with tensor values.
            - queue_names: Names for each tensor.
            - queue_types: Types for each tensor.
        """
        # We parse variable length features (bboxes in a image) as sequence
        # features
        context_example, sequence_example = tf.parse_single_sequence_example(
            record,
            context_features=self.CONTEXT_FEATURES,
            sequence_features=self.SEQUENCE_FEATURES
        )

        # Decode image
        image_raw = tf.image.decode_image(
            context_example['image_raw'], channels=3
        )

        image = tf.cast(image_raw, tf.float32)

        height = tf.cast(context_example['height'], tf.int32)
        width = tf.cast(context_example['width'], tf.int32)
        image_shape = tf.stack([height, width, 3])
        image = tf.reshape(image, image_shape)

        label = self._sparse_to_tensor(sequence_example['label'])
        xmin = self._sparse_to_tensor(sequence_example['xmin'])
        xmax = self._sparse_to_tensor(sequence_example['xmax'])
        ymin = self._sparse_to_tensor(sequence_example['ymin'])
        ymax = self._sparse_to_tensor(sequence_example['ymax'])

        # Stack parsed tensors to define bounding boxes of shape (num_boxes, 5)
        bboxes = tf.stack([xmin, ymin, xmax, ymax, label], axis=1)

        image, bboxes, preprocessing_details = self.preprocess(image, bboxes)

        filename = tf.cast(context_example['filename'], tf.string)

        # TODO: Send additional metadata through the queue (scale_factor,
        # applied_augmentations)

        queue_dtypes = [tf.float32, tf.int32, tf.string, tf.float32]
        queue_names = ['image', 'bboxes', 'filename', 'scale_factor']
        queue_values = {
            'image': image,
            'bboxes': bboxes,
            'filename': filename,
            'scale_factor': preprocessing_details['scale_factor'],
        }

        return queue_values, queue_dtypes, queue_names

    def _augment(self, image, bboxes=None, default_prob=0.5):
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

            augmented = aug_fn(image, bboxes, **aug_config)

            image = tf.cond(
                apply_aug_strategy,
                lambda: augmented['image'],
                lambda: image
            )

            if bboxes is not None:
                bboxes = tf.cond(
                    apply_aug_strategy,
                    lambda: augmented.get('bboxes'),
                    lambda: bboxes
                )

            applied_data_augmentation.append({aug_type: apply_aug_strategy})

        return image, bboxes, applied_data_augmentation

    def _resize_image(self, image, bboxes=None):
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

        return resized['image'], resized.get('bboxes'), resized['scale_factor']

    def _sparse_to_tensor(self, sparse_tensor, dtype=tf.int32, axis=[1]):
        return tf.squeeze(
            tf.cast(tf.sparse_tensor_to_dense(sparse_tensor), dtype), axis=axis
        )
