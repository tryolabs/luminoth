# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def bbox_overlap_tf(bboxes1, bboxes2):
    """Calculate Intersection over Union (IoU) between two sets of bounding
    boxes.

    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

        p1 *-----
           |     |
           |_____* p2

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
    with tf.name_scope('bbox_overlap'):
        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

        xI1 = tf.maximum(x11, tf.transpose(x21))
        yI1 = tf.maximum(y11, tf.transpose(y21))

        xI2 = tf.minimum(x12, tf.transpose(x22))
        yI2 = tf.minimum(y12, tf.transpose(y22))

        intersection = (
            tf.maximum(xI2 - xI1 + 1., 0.) *
            tf.maximum(yI2 - yI1 + 1., 0.)
        )

        bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
        bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

        union = (bboxes1_area + tf.transpose(bboxes2_area)) - intersection

        iou = tf.maximum(intersection / union, 0)

        return iou


def bbox_overlap(bboxes1, bboxes2):
    """Calculate Intersection of Union between two sets of bounding boxes.

    Intersection over Union (IoU) of two bounding boxes A and B is calculated
    doing: (A ∩ B) / (A ∪ B).

    Args:
        bboxes1: numpy array of shape (total_bboxes1, 4).
        bboxes2: numpy array of shape (total_bboxes2, 4).

    Returns:
        iou: numpy array of shape (total_bboxes1, total_bboxes1) a matrix with
            the intersection over union of bboxes1[i] and bboxes2[j] in
            iou[i][j].
    """
    xI1 = np.maximum(bboxes1[:, [0]], bboxes2[:, [0]].T)
    yI1 = np.maximum(bboxes1[:, [1]], bboxes2[:, [1]].T)

    xI2 = np.minimum(bboxes1[:, [2]], bboxes2[:, [2]].T)
    yI2 = np.minimum(bboxes1[:, [3]], bboxes2[:, [3]].T)

    intersection = (
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
    union = (bboxes1_area + bboxes2_area.T) - intersection

    # We start we an empty array of zeros.
    iou = np.zeros((bboxes1.shape[0], bboxes2.shape[0]))

    # Only divide where the intersection is > 0
    np.divide(intersection, union, out=iou, where=intersection > 0.)
    return iou
