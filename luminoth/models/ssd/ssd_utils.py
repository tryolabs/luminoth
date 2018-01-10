import numpy as np
import tensorflow as tf


def adjust_bboxes(bboxes, old_height, old_width, new_height, new_width):
    """Adjusts the bboxes of an image that has been resized.

    Args:
        bboxes: Tensor with shape (num_bboxes, 4).
        old_height: Float. Height of the original image.
        old_width: Float. Width of the original image.
        new_height: Float. Height of the image after resizing.
        new_width: Float. Width of the image after resizing.
    Returns:
        Tensor with shape (num_bboxes, 4), with the adjusted bboxes.
    """
    # We normalize bounding boxes points.
    bboxes_float = tf.to_float(bboxes)
    x_min, y_min, x_max, y_max = tf.unstack(bboxes_float, axis=1)

    x_min = x_min / old_width
    y_min = y_min / old_height
    x_max = x_max / old_width
    y_max = y_max / old_height

    # Use new size to scale back the bboxes points to absolute values.
    x_min = tf.to_int32(x_min * new_width)
    y_min = tf.to_int32(y_min * new_height)
    x_max = tf.to_int32(x_max * new_width)
    y_max = tf.to_int32(y_max * new_height)

    # Concat points and label to return a [num_bboxes, 4] tensor.
    return tf.stack([x_min, y_min, x_max, y_max], axis=1)


def generate_anchors_reference(ratios, scales, num_anchors, feature_map_shape):
    """
    For each ratio we will get an anchor TODO
    Args: TODO
    Returns: convention (x_min, y_min, x_max, y_max) TODO
    """
    heights = np.zeros(num_anchors)
    widths = np.zeros(num_anchors)
    # Because the ratio of 1 we will use the scale sqrt(scale[i] * scale[i+1])
    #  or sqrt(scale[i_max] * scale[i_max]). So we will have to use just
    # `num_anchors` - 1 ratios to generate the anchors
    if len(scales) > 1:
        widths[0] = heights[0] = (np.sqrt(scales[0] * scales[1]) *
                                  feature_map_shape[0])
    # The last endpoint
    else:
        # The scale of the last layer is 0.99, so the extra scale
        # I'll try here will actually be smaller. This is not
        # detailed in the paper and is very hacky, but it makes
        # sense. So it works as a temporary measure.
        heights[0] = scales[0] * feature_map_shape[0] * 0.8
        widths[0] = scales[0] * feature_map_shape[1] * 0.8

    ratios = ratios[:num_anchors - 1]
    heights[1:] = scales[0] / np.sqrt(ratios) * feature_map_shape[0]
    widths[1:] = scales[0] * np.sqrt(ratios) * feature_map_shape[1]

    # Each feature layer forms a grid on image space, so we
    # calculate the center point on the first cell of this grid.
    # Which we'll use as the center for our anchor reference.
    # The center will be the midpoint of the top left cell,
    # given that each cell is of 1x1 size, its center will be 0.5x0.5
    x_center = y_center = 0.5

    # Create anchor reference.
    anchors = np.column_stack([
        x_center - widths / 2,
        y_center - heights / 2,
        x_center + widths / 2,
        y_center + heights / 2,
    ])

    return anchors
