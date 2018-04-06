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
    # x_min, y_min, x_max, y_max = np.split(bboxes, 4, axis=1)
    x_min = bboxes[:, 0] / old_width
    y_min = bboxes[:, 1] / old_height
    x_max = bboxes[:, 2] / old_width
    y_max = bboxes[:, 3] / old_height

    # Use new size to scale back the bboxes points to absolute values.
    x_min = x_min * new_width
    y_min = y_min * new_height
    x_max = x_max * new_width
    y_max = y_max * new_height

    # Concat points and label to return a [num_bboxes, 4] tensor.
    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def generate_anchors_reference(ratios, scales, num_anchors, feature_map_shape):
    """
    Generate the default anchor for one feat map which we will later convolve
    to generate all the anchors of that feat map.
    """
    heights = np.zeros(num_anchors)
    widths = np.zeros(num_anchors)

    if len(scales) > 1:
        widths[0] = heights[0] = (np.sqrt(scales[0] * scales[1]) *
                                  feature_map_shape[0])
    # The last endpoint
    else:
        # The last layer doesn't have a subsequent layer with which
        # to generate the second scale from their geometric mean,
        # so we hard code it to 0.99.
        # We should add this parameter to the config eventually.
        heights[0] = scales[0] * feature_map_shape[0] * 0.99
        widths[0] = scales[0] * feature_map_shape[1] * 0.99

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


def generate_raw_anchors(feature_maps, anchor_min_scale, anchor_max_scale,
                         anchor_ratios, anchors_per_point):
    """
    Returns a dictionary containing the anchors per feature map.

    Returns:
    anchors: A dictionary with feature maps as keys and an array of anchors
        as values ('[[x_min, y_min, x_max, y_max], ...]') with shape
        (anchors_per_point[i] * endpoints_outputs[i][0]
         * endpoints_outputs[i][1], 4)
    """
    # TODO: Anchor generation needs heavy refactor

    # We interpolate the scales of the anchors from a min and a max scale
    scales = np.linspace(anchor_min_scale, anchor_max_scale, len(feature_maps))

    anchors = {}
    for i, (feat_map_name, feat_map) in enumerate(feature_maps.items()):
        feat_map_shape = feat_map.shape.as_list()[1:3]
        anchor_reference = generate_anchors_reference(
            anchor_ratios, scales[i: i + 2],
            anchors_per_point[i], feat_map_shape
        )
        anchors[feat_map_name] = generate_anchors_per_feat_map(
            feat_map_shape, anchor_reference)

    return anchors


def generate_anchors_per_feat_map(feature_map_shape, anchor_reference):
    """Generate anchor for an image.

    Using the feature map, the output of the pretrained network for an
    image, and the anchor_reference generated using the anchor config
    values. We generate a list of anchors.

    Anchors are just fixed bounding boxes of different ratios and sizes
    that are uniformly generated throught the image.

    Args:
        feature_map_shape: Shape of the convolutional feature map used as
            input for the RPN. Should be (batch, height, width, depth).

    Returns:
        all_anchors: A flattened Tensor with all the anchors of shape
            `(num_anchors_per_points * feature_width * feature_height, 4)`
            using the (x1, y1, x2, y2) convention.
    """
    with tf.variable_scope('generate_anchors'):
        shift_x = np.arange(feature_map_shape[1])
        shift_y = np.arange(feature_map_shape[0])
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shift_x = np.reshape(shift_x, [-1])
        shift_y = np.reshape(shift_y, [-1])

        shifts = np.stack(
            [shift_x, shift_y, shift_x, shift_y],
            axis=0
        )

        shifts = np.transpose(shifts)
        # Shifts now is a (H x W, 4) Tensor

        # Expand dims to use broadcasting sum.
        all_anchors = (
            np.expand_dims(anchor_reference, axis=0) +
            np.expand_dims(shifts, axis=1)
        )
        # Flatten
        return np.reshape(all_anchors, (-1, 4))
