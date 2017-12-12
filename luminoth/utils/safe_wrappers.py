import tensorflow as tf


def safe_softmax_cross_entropy_with_logits(
   labels, logits, name='safe_cross_entropy'):
    with tf.name_scope(name):
        safety_condition = tf.greater(
            tf.shape(logits)[0], 0, name='safety_condition'
        )
        return tf.cond(
            safety_condition,
            true_fn=lambda: tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            ),
            false_fn=lambda: tf.constant([], dtype=logits.dtype)
        )
