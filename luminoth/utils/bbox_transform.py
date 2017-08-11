import numpy as np


def get_bbox_properties(bboxes):
    """Get bounding boxes width, height and center point.

    Args:
        bboxes: Numpy array with bounding boxes of shape (total_boxes, 4).

    Returns:
        widths: Numpy array with the width of each bbox.
        heights: Numpy array with the height of each bbox.
        center_x: X-coordinate for center point of each bbox.
        center_y: Y-coordinate for center point of each bbox.
    """
    bboxes_widths = bboxes[:, 2] - bboxes[:, 0] + 1.0
    bboxes_heights = bboxes[:, 3] - bboxes[:, 1] + 1.0
    bboxes_center_x = bboxes[:, 0] + 0.5 * bboxes_widths
    bboxes_center_y = bboxes[:, 1] + 0.5 * bboxes_heights
    return bboxes_widths, bboxes_heights, bboxes_center_x, bboxes_center_y


def encode(proposals, gt_boxes):
    """Encode the different adjustments needed to transform it to its
    corresponding ground truth box.

    Args:
        proposals: Numpy array of shape (total_proposals, 4). Having the
            bbox encoding in the (x_min, y_min, x_max, y_max) order.
        gt_boxes: Numpy array of shape (total_proposals, 4). With the same
            bbox encoding.

    Returns:
        targets: Numpy array of shape (total_proposals, 4) with the different
            values needed to transform the proposal to the gt_boxes.
    """

    (proposal_widths, proposal_heights,
     proposal_center_x, proposal_center_y) = get_bbox_properties(proposals)
    (gt_widths, gt_heights,
     gt_center_x, gt_center_y) = get_bbox_properties(gt_boxes)

    # We need to apply targets as specified by the paper parametrization
    # Faster RCNN 3.1.2
    targets_x = (gt_center_x - proposal_center_x) / proposal_widths
    targets_y = (gt_center_y - proposal_center_y) / proposal_heights
    targets_w = np.log(gt_widths / proposal_widths)
    targets_h = np.log(gt_heights / proposal_heights)

    targets = np.column_stack((targets_x, targets_y, targets_w, targets_h))

    return targets


def decode(bboxes, deltas):
    """
    Args:
        boxes: numpy array of bounding boxes of shape: (num_boxes, 4) following
            the encoding (x_min, y_min, x_max, y_max).
        deltas: numpy array of bounding box deltas, one for each bounding box.
            Its shape is (num_boxes, 4), where the deltas are encoded as
            (dx, dy, dw, dh).

    Returns:
        bboxes: bounding boxes transformed to (x1, y1, x2, y2) coordinates. It
            has the same shape as bboxes.
    """
    widths, heights, ctr_x, ctr_y = get_bbox_properties(bboxes)

    # The dx, dy deltas are relative while the dw, dh deltas are "log relative"
    # d[:, x::y] is used for having a `(num_boxes, 1)` shape instead of
    # `(num_boxes,)`

    # Split deltas columns into flat array
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    # We get the center of the real box as center anchor + relative width
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y

    # New width and height using exp
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    # Calculate (x_min, y_min, x_max, y_max) and pack them together.
    pred_boxes = np.column_stack((
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w - 1.0,
        pred_ctr_y + 0.5 * pred_h - 1.0,
    ))

    return pred_boxes


def clip_points(points, max_val, min_val):
    return np.maximum(np.minimum(points, max_val), min_val)


def clip_boxes(boxes, image_shape):
    """Clip boxes to image boundaries.

    Args:
        boxes: A numpy array of bounding boxes.
        image_shape: Image shape (height, width).
    """
    max_width = image_shape[1] - 1
    max_height = image_shape[0] - 1
    min_width = 0
    min_height = 0

    boxes[:, 0] = clip_points(boxes[:, 0], max_width, min_width)
    boxes[:, 1] = clip_points(boxes[:, 1], max_height, min_height)
    boxes[:, 2] = clip_points(boxes[:, 2], max_width, min_width)
    boxes[:, 3] = clip_points(boxes[:, 3], max_height, min_height)

    return boxes


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)
    """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
