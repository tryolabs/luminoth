import tensorflow as tf


def resize_image(image, bboxes=None, min_size=None, max_size=None):
    """
    We need to resize image and (optionally) bounding boxes when the biggest
    side dimension is bigger than `max_size` or when the smaller side is
    smaller than `min_size`. If no max_size defined it won't scale down and if
    no min_size defined it won't scale up.

    Then, using the ratio we used, we need to properly scale the bounding
    boxes.

    Args:
        image: Tensor with image of shape (H, W, 3).
        bboxes: Optional Tensor with bounding boxes with shape (num_bboxes, 5).
            where we have (x_min, y_min, x_max, y_max, label) for each one.
        min_size: Min size of width or height.
        max_size: Max size of width or height.

    Returns:
        Dictionary containing:
            image: Tensor with scaled image.
            bboxes: Tensor with scaled (using the same factor as the image)
                bounding boxes with shape (num_bboxes, 5).
            scale_factor: Scale factor used to modify the image (1.0 means no
                change).
    """
    image_shape = tf.to_float(tf.shape(image))
    height = image_shape[0]
    width = image_shape[1]

    if min_size:
        # We calculate the upscale factor, the rate we need to use to end up
        # with an image with it's lowest dimension at least `image_min_size`.
        # In case of being big enough the scale factor is 1. (no change)
        min_dimension = tf.minimum(height, width)
        image_min_size = tf.constant(min_size, tf.float32)
        upscale_factor = tf.maximum(image_min_size / min_dimension, 1.)
    else:
        upscale_factor = 1.

    if max_size:
        # We do the same calculating the downscale factor, to end up with an
        # image where the biggest dimension is less than `image_max_size`.
        # When the image is small enough the scale factor is 1. (no change)
        max_dimension = tf.maximum(height, width)
        image_max_size = tf.constant(max_size, tf.float32)
        downscale_factor = tf.minimum(image_max_size / max_dimension, 1.)
    else:
        downscale_factor = 1.

    scale_factor = upscale_factor * downscale_factor

    # New size is calculate using the scale factor and rounding to int.
    new_height = tf.to_int32(height * scale_factor)
    new_width = tf.to_int32(width * scale_factor)

    # Resize image using TensorFlow's own `resize_image` utility.
    image = tf.image.resize_images(
        image, tf.stack([new_height, new_width]),
        method=tf.image.ResizeMethod.BILINEAR
    )

    if bboxes is not None:
        # We normalize bounding boxes points before modifying the image.
        bboxes_float = tf.to_float(bboxes)
        x_min, y_min, x_max, y_max, label = tf.split(bboxes_float, 5, axis=1)

        x_min = x_min / width
        y_min = y_min / height
        x_max = x_max / width
        y_max = y_max / height

        # Use new size to scale back the bboxes points to absolute values.
        x_min = tf.to_int32(x_min * new_width)
        y_min = tf.to_int32(y_min * new_height)
        x_max = tf.to_int32(x_max * new_width)
        y_max = tf.to_int32(y_max * new_height)
        label = tf.to_int32(label)  # Cast back to int.

        # Concat points and label to return a [num_bboxes, 5] tensor.
        bboxes = tf.concat([x_min, y_min, x_max, y_max, label], axis=1)
        return {
            'image': image,
            'bboxes': bboxes,
            'scale_factor': scale_factor,
        }

    return {
        'image': image,
        'scale_factor': scale_factor,
    }
