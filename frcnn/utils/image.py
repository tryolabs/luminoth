import tensorflow as tf


def normalize_bboxes(image, bboxes):
    batch, x1, y1, x2, y2 = tf.unstack(bboxes, axis=1)

    image_shape = tf.cast(tf.shape(image), tf.float32)
    x1 = x1 / image_shape[2]
    y1 = y1 / image_shape[1]
    x2 = x2 / image_shape[2]
    y2 = y2 / image_shape[1]

    bboxes = tf.stack([batch, y1, x1, y2, x2], axis=1)
    bboxes = tf.expand_dims(bboxes, 0)
    return bboxes


def draw_bboxes(image, bboxes, topn=10, normalize=True):
    bboxes = tf.slice(bboxes, [0, 0], [topn, 5])
    if normalize:
        bboxes = normalize_bboxes(image, bboxes)

    return tf.image.draw_bounding_boxes(image, bboxes)
