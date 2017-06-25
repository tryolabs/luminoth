import tensorflow as tf


def draw_bboxes(image, bboxes, topn=10, normalize=False):
    # change fucking order
    #. we asume bboxes has batch
    bboxes = tf.slice(bboxes, [0, 0], [topn, 5])
    batch, x1, y1, x2, y2 = tf.split(value=bboxes, num_or_size_splits=5, axis=1)

    if normalize:
        x1 = x1 / tf.cast(tf.shape(image)[2], tf.float32)
        y1 = y1 / tf.cast(tf.shape(image)[1], tf.float32)
        x2 = x2 / tf.cast(tf.shape(image)[2], tf.float32)
        y2 = y2 / tf.cast(tf.shape(image)[1], tf.float32)

    bboxes = tf.concat([batch, y1, x1, y2, x2], axis=1)
    bboxes = tf.expand_dims(bboxes, 0)
    return tf.image.draw_bounding_boxes(image, bboxes)
