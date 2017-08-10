import numpy as np


def generate_anchors(anchors_reference, anchor_stride, feature_map_size):
    """
    Generate anchors from an anchor_reference using the anchor_stride for an
    image with a feature map size of `feature_map_size`.

    This code is based on the TensorFlow code for generating the same thing
    on the computation graph.

    Args:
        anchors_reference (np.array): with shape (total_anchors, 4), the
            relative distance between the center and the top left X,Y and
            bottom right X, Y of the anchor.
        anchor_stride (int): stride for generation of anchors.
        feature_map_size (np.array): with shape (2,)

    Returns:
        anchors (np.array): array with anchors.
            with shape (height_feature * width_feature * total_anchors, 4)

    TODO: We should create a test for comparing this function vs the one
          actually used in the computation graph.
    """

    grid_width = feature_map_size[1]
    grid_height = feature_map_size[0]

    shift_x = np.arange(grid_width) * anchor_stride
    shift_y = np.arange(grid_height) * anchor_stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = shift_x.reshape([-1])
    shift_y = shift_y.reshape([-1])

    shifts = np.stack(
        [shift_x, shift_y, shift_x, shift_y],
        axis=0
    )

    shifts = shifts.T

    num_anchors = anchors_reference.shape[0]
    num_anchor_points = shifts.shape[0]

    all_anchors = (
        anchors_reference.reshape((1, num_anchors, 4)) +
        np.transpose(
            shifts.reshape((1, num_anchor_points, 4)),
            axes=(1, 0, 2)
        )
    )

    all_anchors = np.reshape(
        all_anchors, (num_anchors * num_anchor_points, 4)
    )

    return all_anchors


if __name__ == '__main__':
    from luminoth.utils.anchors import generate_anchors_reference

    ref = generate_anchors_reference(
        base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)
    )
