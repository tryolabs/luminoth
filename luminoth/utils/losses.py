import tensorflow as tf


def focal_loss(cls_scores, cls_probs, targets, num_classes, gamma=2.0,
               weights=None):
    """Compute RetinaNet's focal loss.

    Args:
        cls_scores: shape (num_proposals, num_classes + 1)
        cls_probs: shape (num_proposals, num_classes + 1)
        targets: shape (num_proposals)
        num_classes: number of classes (not counting background)
        gamma: gamma parameter for focal loss.
        weights: 1D tensor with weights for each class. If set to None,
            all weights will be 1.
    """
    with tf.name_scope('focal_loss'):
        if weights is None:
            weights = 1.
        targets_one_hot = tf.one_hot(
            tf.cast(targets, tf.int32),
            depth=num_classes + 1,
            name='one_hot_targets'
        )
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=targets_one_hot, logits=cls_scores,
            name='compute_cross_entropy'
        )
        weighted_cross_entropy = tf.multiply(
            cross_entropy, weights, name='apply_weights'
        )
        focal_weights = tf.pow(1. - cls_probs, gamma, name='power_gamma')

        focal_loss = tf.multiply(
            focal_weights, weighted_cross_entropy,
            name='apply_gamma_focus'
        )
        return focal_loss


def smooth_l1_loss(bbox_prediction, bbox_target, sigma=1.0):
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
            abs_diff_lt_sigma2, 0.5 * tf.square(abs_diff),
            abs_diff - 0.5
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
