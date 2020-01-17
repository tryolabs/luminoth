import tensorflow as tf
from luminoth.utils.bbox_transform_tf import clip_boxes


def adjust_bboxes(bboxes, old_height, old_width, new_height, new_width):
    """Adjusts the bboxes of an image that has been resized.

    Args:
        bboxes: Tensor with shape (num_bboxes, 5). Last element is the label.
        old_height: Float. Height of the original image.
        old_width: Float. Width of the original image.
        new_height: Float. Height of the image after resizing.
        new_width: Float. Width of the image after resizing.
    Returns:
        Tensor with shape (num_bboxes, 5), with the adjusted bboxes.
    """
    # We normalize bounding boxes points.
    bboxes_float = tf.to_float(bboxes)
    x_min, y_min, x_max, y_max, label = tf.unstack(bboxes_float, axis=1)

    x_min = x_min / old_width
    y_min = y_min / old_height
    x_max = x_max / old_width
    y_max = y_max / old_height

    # Use new size to scale back the bboxes points to absolute values.
    x_min = tf.to_int32(x_min * new_width)
    y_min = tf.to_int32(y_min * new_height)
    x_max = tf.to_int32(x_max * new_width)
    y_max = tf.to_int32(y_max * new_height)
    label = tf.to_int32(label)  # Cast back to int.

    # Concat points and label to return a [num_bboxes, 5] tensor.
    return tf.stack([x_min, y_min, x_max, y_max, label], axis=1)


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

    if max_size is not None:
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
        bboxes = adjust_bboxes(
            bboxes,
            old_height=height, old_width=width,
            new_height=new_height, new_width=new_width
        )
        return {
            'image': image,
            'bboxes': bboxes,
            'scale_factor': scale_factor,
        }

    return {
        'image': image,
        'scale_factor': scale_factor,
    }


def resize_image_fixed(image, new_height, new_width, bboxes=None):

    image_shape = tf.to_float(tf.shape(image))
    height = image_shape[0]
    width = image_shape[1]

    scale_factor_height = new_height / height
    scale_factor_width = new_width / width

    # Resize image using TensorFlow's own `resize_image` utility.
    image = tf.image.resize_images(
        image, tf.stack(tf.to_int32([new_height, new_width])),
        method=tf.image.ResizeMethod.BILINEAR
    )

    if bboxes is not None:
        bboxes = adjust_bboxes(
            bboxes,
            old_height=height, old_width=width,
            new_height=new_height, new_width=new_width
        )
        return {
            'image': image,
            'bboxes': bboxes,
            'scale_factor': (scale_factor_height, scale_factor_width),
        }

    return {
        'image': image,
        'scale_factor': (scale_factor_height, scale_factor_width),
    }


def patch_image(image, bboxes=None, offset_height=0, offset_width=0,
                target_height=None, target_width=None):
    """Gets a patch using tf.image.crop_to_bounding_box and adjusts bboxes

    If patching would leave us with zero bboxes, we return the image and bboxes
    unchanged.

    Args:
        image: Float32 Tensor with shape (H, W, 3).
        bboxes: Tensor with the ground-truth boxes. Shaped (total_boxes, 5).
            The last element in each box is the category label.
        offset_height: Height of the upper-left corner of the patch with
            respect to the original image. Non-negative.
        offset_width: Width of the upper-left corner of the patch with respect
            to the original image. Non-negative.
        target_height: Height of the patch. If set to none, it will be the
            maximum (tf.shape(image)[0] - offset_height - 1). Positive.
        target_width: Width of the patch. If set to none, it will be the
            maximum (tf.shape(image)[1] - offset_width - 1). Positive.

    Returns:
        image: Patch of the original image.
        bboxes: Adjusted bboxes (only those whose centers are inside the
            patch). The key isn't set if bboxes is None.
    """
    # TODO: make this function safe with respect to senseless inputs (i.e
    # having an offset_height that's larger than tf.shape(image)[0], etc.)
    # As of now we only use it inside random_patch, which already makes sure
    # the arguments are legal.
    im_shape = tf.shape(image)
    if target_height is None:
        target_height = (im_shape[0] - offset_height - 1)
    if target_width is None:
        target_width = (im_shape[1] - offset_width - 1)

    new_image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height, offset_width=offset_width,
        target_height=target_height, target_width=target_width
    )
    patch_shape = tf.shape(new_image)

    # Return if we didn't have bboxes.
    if bboxes is None:
        # Resize the patch to the original image's size. This is to make sure
        # we respect restrictions in image size in the models.
        new_image_resized = tf.image.resize_images(
            new_image, im_shape[:2],
            method=tf.image.ResizeMethod.BILINEAR
        )
        return_dict = {'image': new_image_resized}
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
            tf.to_int32(
                clip_boxes(
                    new_bboxes_unclipped,
                    imshape=patch_shape[:2]
                ),
            ),
            masked_bboxes[:, 4:]
        ],
        axis=1
    )
    # Now resize the image to the original size and adjust bboxes accordingly
    new_image_resized = tf.image.resize_images(
        new_image, im_shape[:2],
        method=tf.image.ResizeMethod.BILINEAR
    )
    # adjust_bboxes requires height and width values with dtype=float32
    new_bboxes_resized = adjust_bboxes(
        new_bboxes,
        old_height=tf.to_float(patch_shape[0]),
        old_width=tf.to_float(patch_shape[1]),
        new_height=tf.to_float(im_shape[0]),
        new_width=tf.to_float(im_shape[1])
    )

    # Finally, set up the return dict, but only update the image and bboxes if
    # our patch has at least one bbox in it.
    update_condition = tf.greater_equal(
        tf.shape(new_bboxes_resized)[0],
        1
    )
    return_dict = {}
    return_dict['image'] = tf.cond(
        update_condition,
        lambda: new_image_resized,
        lambda: image
    )
    return_dict['bboxes'] = tf.cond(
        update_condition,
        lambda: new_bboxes_resized,
        lambda: bboxes
    )
    return return_dict


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


def random_patch(image, bboxes=None, min_height=600, min_width=600,
                 seed=None):
    """Gets a random patch from an image.

    min_height and min_width values will be normalized if they are not possible
    given the input image's shape. See also patch_image.

    Args:
        image: Tensor with shape (H, W, 3).
        bboxes: Tensor with the ground-truth boxes. Shaped (total_boxes, 5).
            The last element in each box is the category label.
        min_height: Minimum height of the patch.
        min_width: Minimum width of the patch.
        seed: Seed to be used in randomizing functions.

    Returns:
        image: Tensor with shape (H', W', 3), with H' <= H and W' <= W. A
            random patch of the input image.
        bboxes: Tensor with shape (new_total_boxes, 5), where we keep
            bboxes that have their center inside the patch, cropping
            them to the patch boundaries. If we didn't get any bboxes, the
            return dict will not have the 'bboxes' key defined.
    """
    # Start by normalizing the arguments.
    # Our patch can't be larger than the original image.
    im_shape = tf.shape(image)
    min_height = tf.minimum(min_height, im_shape[0] - 1)
    min_width = tf.minimum(min_width, im_shape[1] - 1)

    # Now get the patch using tf.image.crop_to_bounding_box.
    # See the documentation on tf.image.crop_to_bounding_box or the explanation
    # in patch_image for the meaning of these variables.
    offset_width = tf.random_uniform(
        shape=[],
        minval=0,
        maxval=tf.subtract(
            im_shape[1],
            min_width
        ),
        dtype=tf.int32,
        seed=seed
    )
    offset_height = tf.random_uniform(
        shape=[],
        minval=0,
        maxval=tf.subtract(
            im_shape[0],
            min_height
        ),
        dtype=tf.int32,
        seed=seed
    )
    target_width = tf.random_uniform(
        shape=[],
        minval=min_width,
        maxval=tf.subtract(
            im_shape[1],
            offset_width
        ),
        dtype=tf.int32,
        seed=seed
    )
    target_height = tf.random_uniform(
        shape=[],
        minval=min_height,
        maxval=tf.subtract(
            im_shape[0],
            offset_height
        ),
        dtype=tf.int32,
        seed=seed
    )
    return patch_image(
        image, bboxes=bboxes,
        offset_height=offset_height, offset_width=offset_width,
        target_height=target_height, target_width=target_width
    )


def random_resize(image, bboxes=None, min_size=600, max_size=980,
                  seed=None):
    """Randomly resizes an image within limits.

    Args:
        image: Tensor with shape (H, W, 3)
        bboxes: Tensor with the ground-truth boxes. Shaped (total_boxes, 5).
            The last element in each box is the category label.
        min_size: minimum side-size of the resized image.
        max_size: maximum side-size of the resized image.
        seed: Seed to be used in randomizing functions.

    Returns:
        image: Tensor with shape (H', W', 3), satisfying the following.
            min_size <= H' <= H
            min_size <= W' <= W
        bboxes: Tensor with the same shape as the input bboxes, if we had them.
            Else, this key will not be set.
    """
    im_shape = tf.to_float(tf.shape(image))
    new_size = tf.random_uniform(
        shape=[2],
        minval=min_size,
        maxval=max_size,
        dtype=tf.int32,
        seed=seed,
    )
    image = tf.image.resize_images(
        image, new_size,
        method=tf.image.ResizeMethod.BILINEAR
    )
    # Our returned dict needs to have a fixed size. So we can't
    # return the scale_factor that resize_image returns.
    if bboxes is not None:
        new_size = tf.to_float(new_size)
        bboxes = adjust_bboxes(
            bboxes,
            old_height=im_shape[0], old_width=im_shape[1],
            new_height=new_size[0], new_width=new_size[1]
        )
        return_dict = {
            'image': image,
            'bboxes': bboxes
        }
    else:
        return_dict = {'image': image}
    return return_dict


def random_distortion(image, bboxes=None, brightness=None, contrast=None,
                      hue=None, saturation=None, seed=None):
    """Photometrically distorts an image.

    This includes changing the brightness, contrast and hue.

    Args:
        image: Tensor with shape (H, W, 3)
        brightness:
            max_delta: non-negative float
        contrast:
            lower: non-negative float
            upper: non-negative float
        hue:
            max_delta: float in [0, 0.5]
        saturation:
            lower: non-negative float
            upper: non-negative float
        seed: Seed to be used in randomizing functions.

    Returns:
        image: Distorted image with the same shape as the input image.
        bboxes: Unchanged bboxes.
    """
    # Following Andrew Howard (2013). "Some improvements on deep convolutional
    # neural network based image classification."
    if brightness is not None:
        if 'max_delta' not in brightness:
            brightness.max_delta = 0.3
        image = tf.image.random_brightness(
            image, max_delta=brightness.max_delta, seed=seed
        )
    # Changing contrast, even with parameters close to 1, can lead to
    # excessively distorted images. Use with care.
    if contrast is not None:
        if 'lower' not in contrast:
            contrast.lower = 0.8
        if 'upper' not in contrast:
            contrast.upper = 1.2
        image = tf.image.random_contrast(
            image, lower=contrast.lower, upper=contrast.upper,
            seed=seed
        )
    if hue is not None:
        if 'max_delta' not in hue:
            hue.max_delta = 0.2
        image = tf.image.random_hue(
            image, max_delta=hue.max_delta, seed=seed
        )
    if saturation is not None:
        if 'lower' not in saturation:
            saturation.lower = 0.8
        if 'upper' not in saturation:
            saturation.upper = 1.2
        image = tf.image.random_saturation(
            image, lower=saturation.lower, upper=saturation.upper,
            seed=seed
        )
    if bboxes is None:
        return_dict = {'image': image}
    else:
        return_dict = {
            'image': image,
            'bboxes': bboxes,
        }
    return return_dict


def expand(image, bboxes=None, fill=0, min_ratio=1, max_ratio=4, seed=None):
    """
    Increases the image size by adding large padding around the image

    Acts as a zoom out of the image, and when the image is later resized to
    the input size the network expects, it provides smaller size object
    examples.

    Args:
        image: Tensor with image of shape (H, W, 3).
        bboxes: Optional Tensor with bounding boxes with shape (num_bboxes, 5).
            where we have (x_min, y_min, x_max, y_max, label) for each one.

    Returns:
        Dictionary containing:
            image: Tensor with zoomed out image.
            bboxes: Tensor with zoomed out bounding boxes with shape
                (num_bboxes, 5).
    """
    image_shape = tf.to_float(tf.shape(image))
    height = image_shape[0]
    width = image_shape[1]
    size_multiplier = tf.random_uniform([1], minval=min_ratio,
                                        maxval=max_ratio, seed=seed)

    # Expand image
    new_height = height * size_multiplier
    new_width = width * size_multiplier
    pad_left = tf.random_uniform([1], minval=0,
                                 maxval=new_width-width, seed=seed)
    pad_right = new_width - width - pad_left
    pad_top = tf.random_uniform([1], minval=0,
                                maxval=new_height-height, seed=seed)
    pad_bottom = new_height - height - pad_top

    # TODO: use mean instead of 0 for filling the paddings
    paddings = tf.stack([tf.concat([pad_top, pad_bottom], axis=0),
                         tf.concat([pad_left, pad_right], axis=0),
                         tf.constant([0., 0.])])
    expanded_image = tf.pad(image, tf.to_int32(paddings), constant_values=fill)

    # Adjust bboxes
    shift_bboxes_by = tf.concat([pad_left, pad_top, pad_left, pad_top], axis=0)
    bbox_labels = tf.reshape(bboxes[:, 4], (-1, 1))
    bbox_adjusted_coords = bboxes[:, :4] + tf.to_int32(shift_bboxes_by)
    bbox_adjusted = tf.concat([bbox_adjusted_coords, bbox_labels], axis=1)

    # Return results
    return_dict = {'image': expanded_image}
    if bboxes is not None:
        return_dict['bboxes'] = bbox_adjusted
    return return_dict


def fisheye(image, bboxes=None, min_top_padding=0.1, max_top_padding=1.0, min_bottom_padding=0.1,
            max_bottom_padding=1.0, min_left_padding=0., max_left_padding=0.5, min_right_padding=0.,
            max_right_padding=0.5):
    """
    Applies a fisheye effect transformation, taking the top-left corner as the fisheye
    center. This generates a quarter-fisheye image.

    The fisheye center can be changed by flipping the image.

    Args:
        image: Tensor with image of shape (H, W, 3).
        bboxes: Optional Tensor with bounding boxes with shape (num_bboxes, 5).
            where we have (x_min, y_min, x_max, y_max, label) for each one.
        min_top_padding: Min fraction of image height to determine 0-padding on top of it.
        max_top_padding: Max fraction of image height to determine 0-padding on top of it.
        min_bottom_padding: Min fraction of image height to determine 0-padding below of it.
        max_bottom_padding: Max fraction of image height to determine 0-padding below of it.
        min_left_padding: Min fraction of image width to determine 0-padding on left side.
        max_left_padding: Max fraction of image widtht to determine 0-padding on left side.
        min_right_padding: Min fraction of image width to determine 0-padding on right side.
        max_right_padding: Max fraction of image width to determine 0-padding on right side.


    Returns:
        Dictionary containing:
            image: Tensor with transformed out image.
            bboxes: Tensor with transformed out bounding boxes with shape
                (num_bboxes, 5).
    """

    def interpolate_bilinear(img, y, x):
        return img[tf.floor(y), tf.floor(x)] * (1 - (y - tf.floor(y))) * (1 - (x - tf.floor(x)))
        + img[tf.floor(y), tf.floor(x) + 1] * (1 - (y - tf.floor(y))) * (x - tf.floor(x))
        + img[tf.floor(y) + 1, tf.floor(x)] * (y - tf.floor(y)) * (1 - (x - tf.floor(x)))
        + img[tf.floor(y) + 1, tf.floor(x) + 1] * (y - tf.floor(y)) * (x - tf.floor(x))

    # Generate padding values
    top_padding = tf.random_uniform([], min_top_padding, max_top_padding)
    bottom_padding = tf.random_uniform([], min_bottom_padding, max_bottom_padding)
    left_padding = tf.random_uniform([], min_left_padding, max_left_padding)
    right_padding = tf.random_uniform([], min_right_padding, max_right_padding)

    image_shape = tf.to_float(tf.shape(image))
    height = image_shape[0]
    width = image_shape[1]

    # Compute absolute length of paddings
    top_padding, bottom_padding, left_padding, right_padding = (
        tf.cast(height * top_padding, tf.int32),
        tf.cast(height * bottom_padding, tf.int32),
        tf.cast(width * left_padding, tf.int32),
        tf.cast(width * right_padding, tf.int32)
    )
    vertical_paddings_tensor = tf.stack([top_padding, bottom_padding])
    horizontal_paddings_tensor = tf.stack([left_padding, right_padding])

    # Apply paddings on top, bottom and sides of image
    image = tf.pad(
        image, tf.stack([vertical_paddings_tensor, horizontal_paddings_tensor, [0, 0]]),
        'constant', constant_values=0
    )

    # Adjust bboxes to paddings
    x_min, y_min, x_max, y_max, label = tf.unstack(bboxes, axis=1)
    y_min += top_padding
    y_max += top_padding
    x_min += left_padding
    x_max += left_padding
    bboxes = tf.stack([x_min, y_min, x_max, y_max, label], axis=1)

    # Recompute image height and width with added padings
    height = tf.cast(height, tf.int32) + top_padding + bottom_padding
    width = width + tf.cast(left_padding + right_padding, tf.float32)

    out_width = out_height = tf.cast(height - bottom_padding, tf.float32)

    # Apply fisheye transformation
    # Map each pixel from source image, taking y as radius and x as cos(theta) times its width.
    X = tf.range(out_width, dtype=tf.float32)
    Y = tf.range(out_height, dtype=tf.float32)

    X_2 = tf.square(X)
    Y_2 = tf.square(Y)
    # Compute radius of each coordinate
    # One-pad vectors to compute sum of squares as a matrix product
    X_2 = tf.stack([X_2, tf.ones(tf.cast(out_width, tf.int32), tf.float32)])
    Y_2 = tf.stack([tf.ones(tf.cast(out_height, tf.int32), tf.float32), Y_2], axis=1)
    yS = tf.sqrt(tf.matmul(Y_2, X_2))

    # Compute angle of each coordinate
    theta = tf.atan(tf.matmul(tf.expand_dims(Y, 1), tf.expand_dims(1/X, 0)))
    xS = tf.cos(theta) * width

    coords = tf.stack([xS, yS], axis=2)
    # Expand image dims since resampler expects shape [batch_size, data_height, data_width,
    # data_channels]
    out_image = tf.contrib.resampler.resampler(tf.expand_dims(image, 0), tf.expand_dims(coords, 0))

    # Transform bboxes. The transformed bboxes are not squares since they get deformed,
    # so we generate square bboxes that contains the transformed ones
    x_min, y_min, x_max, y_max, label = tf.unstack(bboxes, axis=1)
    x_min, y_min, x_max, y_max = [tf.cast(x, tf.float32) for x in [x_min, y_min, x_max, y_max]]

    # Apply transformation to bboxes, getting polar coordinates
    r_min_dst = y_min
    r_max_dst = y_max
    theta_min_dst = tf.acos(x_min / width)
    theta_max_dst = tf.acos(x_max / width)

    # Transform to cartesian
    x_min_dst = r_min_dst * (x_min / width)  # Simplify r_min_dst * tf.cos(tf.acos(x_min / width))
    x_max_dst = r_max_dst * (x_max / width)
    y_min_dst = r_min_dst * tf.sin(theta_max_dst)
    y_max_dst = r_max_dst * tf.sin(theta_min_dst)

    x_min, y_min, x_max, y_max = [tf.cast(x, tf.int32) for x in [x_min_dst,
                                  y_min_dst, x_max_dst, y_max_dst]]
    bboxes = tf.stack([x_min, y_min, x_max, y_max, label], axis=1)

    return_dict = {'image': tf.reshape(
        out_image, [height - bottom_padding, height - bottom_padding, 3]
        )
    }
    if bboxes is not None:
        return_dict['bboxes'] = bboxes
    return return_dict
