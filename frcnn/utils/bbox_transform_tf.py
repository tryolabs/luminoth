import tensorflow as tf


def get_width_upright(bboxes):
    with tf.name_scope('get_width_upright'):
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        width = x2 - x1 + 1.
        height = y2 - y1 + 1.

        # Calculate up right point of bbox (urx = up right x)
        urx = x1 + .5 * width
        ury = y1 + .5 * height

        return width, height, urx, ury


def bbox_encode(bboxes, gt_boxes):
    with tf.name_scope('bbox_encode'):
        (bboxes_width, bboxes_height,
         bboxes_urx, bboxes_ury) = get_width_upright(bboxes)

        (gt_boxes_width, gt_boxes_height,
         gt_boxes_urx, gt_boxes_ury) = get_width_upright(gt_boxes)

        targets_dx = (gt_boxes_urx - bboxes_urx) / bboxes_width
        targets_dy = (gt_boxes_ury - bboxes_ury) / bboxes_height

        targets_dw = tf.log(gt_boxes_width / bboxes_width)
        targets_dh = tf.log(gt_boxes_height / bboxes_height)

        targets = tf.concat(
            [targets_dx, targets_dy, targets_dw, targets_dh], axis=1)

        return targets


def bbox_decode(roi, deltas):
    with tf.name_scope('bbox_decode'):
        (roi_width, roi_height,
         roi_urx, roi_ury) = get_width_upright(roi)

        dx, dy, dw, dh = tf.split(deltas, 4, axis=1)

        pred_ur_x = dx * roi_width + roi_urx
        pred_ur_y = dy * roi_height + roi_ury
        pred_w = tf.exp(dw) * roi_width
        pred_h = tf.exp(dh) * roi_height

        bbox_x1 = pred_ur_x - 0.5 * pred_w
        bbox_y1 = pred_ur_y - 0.5 * pred_h

        # This -1. extra is different from reference implementation.
        bbox_x2 = pred_ur_x + 0.5 * pred_w - 1.
        bbox_y2 = pred_ur_y + 0.5 * pred_h - 1.

        bboxes = tf.concat(
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=1)

        return bboxes


def clip_bboxes(bboxes, imshape):
    """
    Clips bounding boxes to image boundaries based on image shape.

    Args:
        bboxes: Tensor with shape (num_bboxes, 4)
            where point order is x1, y1, x2, y2.

        imshape: Tensor with shape (2, )
            where the first value is height and the next is width.

    Returns
        Tensor with same shape as bboxes but making sure that none
        of the bboxes are inside the image.
    """
    with tf.name_scope('clip_bboxes'):
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        imshape = tf.cast(imshape, dtype=tf.float32)
        width = imshape[1]
        height = imshape[0]
        x1 = tf.maximum(tf.minimum(x1, width - 1.), 0)
        x2 = tf.maximum(tf.minimum(x2, width - 1.), 0)

        y1 = tf.maximum(tf.minimum(y1, height - 1.), 0)
        y2 = tf.maximum(tf.minimum(y2, height - 1.), 0)

        bboxes = tf.concat([x1, y1, x2, y2], axis=1)

        return bboxes

if __name__ == '__main__':
    import numpy as np

    bboxes = tf.placeholder(tf.float32)
    bboxes_val = [[10, 10, 20, 22]]

    gt_boxes = tf.placeholder(tf.float32)
    gt_boxes_val = [[11, 13, 34, 31]]

    imshape = tf.placeholder(tf.int32)
    imshape_val = (100, 100)

    deltas = bbox_encode(bboxes, gt_boxes)
    decoded_bboxes = bbox_decode(bboxes, deltas)
    final_decoded_bboxes = clip_bboxes(decoded_bboxes, imshape)

    with tf.Session() as sess:
        final_decoded_bboxes = sess.run(final_decoded_bboxes, feed_dict={
            bboxes: bboxes_val,
            gt_boxes: gt_boxes_val,
            imshape: imshape_val,
        })

        assert np.all(gt_boxes_val == final_decoded_bboxes)

        print(final_decoded_bboxes)

