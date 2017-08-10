import numpy as np


def generate_gt_boxes(total_boxes, image_size, min_size=10,
                      total_classes=None):
    """
    Generate `total_boxes` fake (but consistent) ground-truth boxes for an
    image of size `image_size` (height, width).

    Args:
        total_boxes (int): The total number of boxes.
        image_size (tuple): Size of the fake image.

    Returns:
        gt_boxes (np.array): With shape [total_boxes, 4].
    """

    image_size = np.array(image_size)

    assert (image_size > min_size).all(), \
        'Can\'t generate gt_boxes that small for that image size'

    # Generate random sizes for each boxes.
    max_size = np.min(image_size) - min_size
    random_sizes = np.random.randint(
        low=min_size, high=max_size,
        size=(total_boxes, 2)
    )

    # Generate random starting points for boundind boxes (left top point)
    random_leftop = np.random.randint(
        low=0, high=max_size, size=(total_boxes, 2)
    )

    rightbottom = np.minimum(
        random_sizes + random_leftop,
        np.array(image_size) - 1
    )

    gt_boxes = np.column_stack((random_leftop, rightbottom))

    # TODO: Remove asserts after writing tests for this function.
    assert (gt_boxes[:, 0] < gt_boxes[:, 2]).all(), \
        'Gt boxes without consistent Xs'
    assert (gt_boxes[:, 1] < gt_boxes[:, 3]).all(), \
        'Gt boxes without consistent Ys'

    if total_classes:
        random_classes = np.random.randint(
            low=0, high=total_classes - 1, size=(total_boxes, 1))
        gt_boxes = np.column_stack((gt_boxes, random_classes))

        assert (gt_boxes[:, 1] < total_classes).all(), \
            'Gt boxes without consistent classes'

    return gt_boxes
