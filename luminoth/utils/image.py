import tensorflow as tf

from utils.bbox_transform_tf import clip_boxes


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

    if min_size is not None:
        # We calculate the upscale factor, the rate we need to use to end up
        # with an image with it's lowest dimension at least `image_min_size`.
        # In case of being big enough the scale factor is 1. (no change)
        min_size = tf.to_float(min_size)
        min_dimension = tf.minimum(height, width)
        upscale_factor = tf.maximum(min_size / min_dimension, 1.)
    else:
        upscale_factor = tf.constant(1.)

    if max_size:
        # We do the same calculating the downscale factor, to end up with an
        # image where the biggest dimension is less than `image_max_size`.
        # When the image is small enough the scale factor is 1. (no change)
        max_size = tf.to_float(max_size)
        max_dimension = tf.maximum(height, width)
        downscale_factor = tf.minimum(max_size / max_dimension, 1.)
    else:
        downscale_factor = tf.constant(1.)

    scale_factor = upscale_factor * downscale_factor

    # New size is calculate using the scale factor and rounding to int.
    new_height = height * scale_factor
    new_width = width * scale_factor

    # Resize image using TensorFlow's own `resize_image` utility.
    image = tf.image.resize_images(
        image, tf.stack(tf.to_int32([new_height, new_width])),
        method=tf.image.ResizeMethod.BILINEAR
    )

    if bboxes is not None:
        # We normalize bounding boxes points before modifying the image.
        bboxes_float = tf.to_float(bboxes)
        x_min, y_min, x_max, y_max, label = tf.unstack(bboxes_float, axis=1)

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
        bboxes = tf.stack([x_min, y_min, x_max, y_max, label], axis=1)
        return {
            'image': image,
            'bboxes': bboxes,
            'scale_factor': scale_factor,
        }

    return {
        'image': image,
        'scale_factor': scale_factor,
    }


def flip_image(image, bboxes=None, left_right=True, up_down=False):
    """Flips image on its axis for data augmentation.

    Args:
        image: Tensor with image of shape (H, W, 3).
        bboxes: Optional Tensor with bounding boxes with shape
            (total_bboxes, 5).
        left_right: Boolean flag to flip the image horizontally
            (left to right).
        up_down: Boolean flag to flip the image vertically (upside down)
    Returns:
        image: Flipped image with the same shape.
        bboxes: Tensor with the same shape.
    """

    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]

    if bboxes is not None:
        # bboxes usually come from dataset as ints, but just in case we are
        # using flip for preprocessing, where bboxes usually are represented as
        # floats, we cast them.
        bboxes = tf.to_int32(bboxes)

    if left_right:
        image = tf.image.flip_left_right(image)
        if bboxes is not None:
            x_min, y_min, x_max, y_max, label = tf.unstack(bboxes, axis=1)
            new_x_min = width - x_max - 1
            new_y_min = y_min
            new_x_max = new_x_min + (x_max - x_min)
            new_y_max = y_max
            bboxes = tf.stack(
                [new_x_min, new_y_min, new_x_max, new_y_max, label], axis=1
            )

    if up_down:
        image = tf.image.flip_up_down(image)
        if bboxes is not None:
            x_min, y_min, x_max, y_max, label = tf.unstack(bboxes, axis=1)
            new_x_min = x_min
            new_y_min = height - y_max - 1
            new_x_max = x_max
            new_y_max = new_y_min + (y_max - y_min)
            bboxes = tf.stack(
                [new_x_min, new_y_min, new_x_max, new_y_max, label], axis=1
            )

    return_dict = {'image': image}
    if bboxes is not None:
        return_dict['bboxes'] = bboxes

    return return_dict


def random_patch(image, bboxes=None, debug=False):
    """Gets a random patch from an image.

    Args:
        image: Tensor with shape (H, W, 3).
        bboxes: Tensor with the ground-truth boxes. Shaped (total_boxes, 5).
            The last element in each box is the category label.

    Returns:
        image: Tensor with shape (H', W', 3), with H' <= H and W' <= W. A
            random patch of the input image.
        bboxes: Tensor with shape (new_total_boxes, 5), where we keep
            bboxes that have their center inside the patch, cropping
            them to the patch boundaries.
    """
    if debug:
        seed = 0
    else:
        seed = None
    # See the documentation on tf.crop_to_bounding_box for the meaning of
    # these variables.
    offset_width = tf.random_uniform(
        shape=[],
        minval=0,
        maxval=tf.subtract(
            tf.shape(image)[1],
            1
        ),
        dtype=tf.int32,
        seed=seed
    )
    offset_height = tf.random_uniform(
        shape=[],
        minval=0,
        maxval=tf.subtract(
            tf.shape(image)[0],
            1
        ),
        dtype=tf.int32,
        seed=seed
    )
    target_width = tf.random_uniform(
        shape=[],
        minval=1,
        maxval=tf.subtract(
            tf.shape(image)[1],
            offset_width
        ),
        dtype=tf.int32,
        seed=seed
    )
    target_height = tf.random_uniform(
        shape=[],
        minval=1,
        maxval=tf.subtract(
            tf.shape(image)[0],
            offset_height
        ),
        dtype=tf.int32,
        seed=seed
    )
    new_image = tf.image.crop_to_bounding_box(
        image,
        offset_height, offset_width,
        target_height, target_width
    )

    return_dict = {'image': new_image}

    # Return if we didn't have bboxes.
    if bboxes is None:
        return_dict['bboxes'] = tf.constant(-1.)
        return return_dict

    # Now we will remove all bboxes whose centers are not inside the cropped
    # image.

    # First get the x  and y coordinates of the center of each of the
    # bboxes.
    bboxes_center_x = tf.reduce_mean(
        tf.concat(
            [
                # bboxes[:, 0] gets a Tensor with shape (20,).
                # We do this to get a Tensor with shape (20, 1).
                bboxes[:, 0:1],
                bboxes[:, 2:3]
            ],
            axis=1
        )
    )
    bboxes_center_y = tf.reduce_mean(
        tf.concat(
            [
                bboxes[:, 1:2],
                bboxes[:, 3:4]
            ],
            axis=1
        ),
        axis=1
    )

    # Now we get a boolean tensor holding for each of the bboxes' centers
    # wheter they are inside the patch.
    center_x_is_inside = tf.logical_and(
        tf.greater(
            bboxes_center_x,
            offset_width
        ),
        tf.less(
            bboxes_center_x,
            tf.add(target_width, offset_width)
        )
    )
    center_y_is_inside = tf.logical_and(
        tf.greater(
            bboxes_center_y,
            offset_height
        ),
        tf.less(
            bboxes_center_y,
            tf.add(target_height, offset_height)
        )
    )
    center_is_inside = tf.logical_and(
        center_x_is_inside,
        center_y_is_inside
    )

    # Now we mask the bboxes, removing all those whose centers are outside
    # the patch.
    masked_bboxes = tf.boolean_mask(bboxes, center_is_inside)
    # We move the bboxes to the right place, clipping them if
    # necessary.
    new_bboxes_unclipped = tf.concat(
        [
            tf.subtract(masked_bboxes[:, 0:1], offset_width),
            tf.subtract(masked_bboxes[:, 1:2], offset_height),
            tf.subtract(masked_bboxes[:, 2:3], offset_width),
            tf.subtract(masked_bboxes[:, 3:4], offset_height),
        ],
        axis=1,
    )
    # Finally, we clip the boxes and add back the labels.
    new_bboxes = tf.concat(
        [
            clip_boxes(
                new_bboxes_unclipped[:, :4],
                imshape=tf.shape(new_image)[:2]
            ),
            tf.cast(masked_bboxes[:, 4:], tf.float32)
        ],
        axis=1
    )
    return_dict['bboxes'] = new_bboxes
    return return_dict
