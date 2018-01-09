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
    if scales.shape[0] > 1:
        widths[0] = heights[0] = (np.sqrt(scales[0] * scales[1]) *
                                  feature_map_shape[0])
    # The last endpoint
    else:
        widths[0] = heights[0] = scales[0]
    ratios = ratios[:num_anchors - 1]
    heights[1:] = scales[0] / np.sqrt(ratios) * feature_map_shape[0]
    widths[1:] = scales[0] * np.sqrt(ratios) * feature_map_shape[1]

    # Center point has the same X, Y value.
    center_xy = 0

    # Create anchor reference.
    anchors = np.column_stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2,
    ])

    return anchors
