import numpy as np


def bbox_overlaps(bboxes1, bboxes2):
    xI1 = np.maximum(bboxes1[:, [0]], bboxes2[:, [0]].T)
    yI1 = np.maximum(bboxes1[:, [1]], bboxes2[:, [1]].T)

    xI2 = np.minimum(bboxes1[:, [2]], bboxes2[:, [2]].T)
    yI2 = np.minimum(bboxes1[:, [3]], bboxes2[:, [3]].T)

    inter_area = (
        np.maximum(xI2 - xI1 + 1, 0.) *
        np.maximum(yI2 - yI1 + 1, 0.)
    )

    bboxes1_area = (
        (bboxes1[:, [2]] - bboxes1[:, [0]] + 1) *
        (bboxes1[:, [3]] - bboxes1[:, [1]] + 1)
    )
    bboxes2_area = (
        (bboxes2[:, [2]] - bboxes2[:, [0]] + 1) *
        (bboxes2[:, [3]] - bboxes2[:, [1]] + 1)
    )

    # Calculate the union as the sum of areas minus intersection
    union = (bboxes1_area + bboxes2_area.T) - inter_area

    # We start we an empty array of zeros.
    iou = np.zeros((bboxes1.shape[0], bboxes2.shape[0]))

    # Only divide where the intersection is > 0
    np.divide(inter_area, union, out=iou, where=inter_area > 0.)
    return iou
