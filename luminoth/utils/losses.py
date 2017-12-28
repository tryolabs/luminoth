import tensorflow as tf


def smooth_l1_loss(bbox_prediction, bbox_target, sigma=3.0):
    """
    Return Smooth L1 Loss for bounding box prediction.

    Args:
        bbox_prediction: shape (1, H, W, num_anchors * 4)
        bbox_target:     shape (1, H, W, num_anchors * 4)


    Smooth L1 loss is defined as:

    0.5 * x^2                  if |x| < d
    abs(x) - 0.5               if |x| >= d

    Where d = 1 and x = prediction - target

    """
    sigma2 = sigma ** 2
    diff = bbox_prediction - bbox_target
    abs_diff = tf.abs(diff)
    abs_diff_lt_sigma2 = tf.less(abs_diff, 1.0 / sigma2)
    bbox_loss = tf.reduce_sum(
        tf.where(
            abs_diff_lt_sigma2,
            0.5 * sigma2 * tf.square(abs_diff),
            abs_diff - 0.5 / sigma2
        ), [1]
    )
    return bbox_loss


if __name__ == '__main__':
    bbox_prediction_tf = tf.placeholder(tf.float32)
    bbox_target_tf = tf.placeholder(tf.float32)
    loss_tf = smooth_l1_loss(bbox_prediction_tf, bbox_target_tf)
    with tf.Session() as sess:
        loss = sess.run(
            loss_tf,
            feed_dict={
                bbox_prediction_tf: [
                    [0.47450006, -0.80413032, -0.26595005, 0.17124325]
                ],
                bbox_target_tf: [
                    [0.10058594, 0.07910156, 0.10555581, -0.1224325]
                ],
            })
