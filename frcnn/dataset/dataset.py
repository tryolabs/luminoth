import sonnet as snt
import tensorflow as tf


class Dataset(snt.AbstractModule):
    def __init__(self, config, **kwargs):
        self._cfg = config
        self._num_classes = kwargs.pop('num_classes', self._cfg.NUM_CLASSES)
        self._dataset_dir = kwargs.pop('dataset_dir', self._cfg.DATASET_DIR)
        self._num_epochs = kwargs.pop('num_epochs', self._cfg.NUM_EPOCHS)
        self._batch_size = kwargs.pop('batch_size', self._cfg.BATCH_SIZE)
        self._subset = kwargs.pop('subset', self._cfg.TRAIN_SUBSET)
        self._image_min_size = kwargs.pop('image_min_size', self._cfg.IMAGE_MIN_SIZE)
        self._image_max_size = kwargs.pop('image_max_size', self._cfg.IMAGE_MAX_SIZE)

        super(Dataset, self).__init__(**kwargs)

    def _resize_image(self, image, bboxes):
        """
        We need to resize image and bounding boxes when the biggest side
        dimension is bigger than `image_max_size` or when the smaller side is
        smaller than `image_min_size`.

        Then, using the ratio we used, we need to properly scale the bounding boxes.

        Args:
            image: Tensor with image of shape (H, W, 3).
            bboxes: Tensor with bounding boxes with shape (num_bboxes, 5).
                where we have (x_min, y_min, x_max, y_max, label) for each one.

        Returns:
            image: Scaled image.
            bboxes: Scaled bboxes.
        """
        image_shape = tf.to_float(tf.shape(image))
        height = image_shape[0]
        width = image_shape[1]

        # We normalize bounding boxes points before modifying the image.
        bboxes_float = tf.to_float(bboxes)
        x_min, y_min, x_max, y_max, label = tf.split(bboxes_float, 5, axis=1)

        x_min = x_min / width
        y_min = y_min / height
        x_max = x_max / width
        y_max = y_max / height


        # We calculate the upscale factor, the rate we need to use to end up
        # with an image with it's lowest dimension at least `image_min_size`.
        # In case of being already big enough the scale factor is 1. (no change)
        min_dimension = tf.minimum(height, width)
        image_min_size = tf.constant(self._image_min_size, tf.float32)
        upscale_factor = tf.maximum(image_min_size / min_dimension, 1.)

        # We do the same calculating the downscale factor, to end up with an image
        # where the biggest dimension is less than `image_max_size`.
        # When the image is small enough the scale factor is 1. (no change)
        max_dimension = tf.maximum(height, width)
        image_max_size = tf.constant(self._image_max_size, tf.float32)
        downscale_factor = tf.minimum(image_max_size / max_dimension, 1.)

        scale_factor = upscale_factor * downscale_factor

        # New size is calculate using the scale factor and rounding to int.
        new_height = height * scale_factor
        new_width = width * scale_factor

        # Use new size to scale back the bboxes points to absolute values.
        x_min = tf.to_int32(x_min * new_width)
        y_min = tf.to_int32(y_min * new_height)
        x_max = tf.to_int32(x_max * new_width)
        y_max = tf.to_int32(y_max * new_height)
        label = tf.to_int32(label)  # Cast back to int.

        new_height = tf.to_int32(new_height)
        new_width = tf.to_int32(new_width)

        image = tf.image.resize_images(
            image, tf.stack([new_height, new_width]),
            method=tf.image.ResizeMethod.BILINEAR
        )

        bboxes = tf.concat([x_min, y_min, x_max, y_max, label], axis=1)

        return image, bboxes, scale_factor
