import tensorflow as tf


def spatial_softmax(input):
    with tf.name_scope('SpatialSoftmax'):
        input_shape = tf.shape(input)
        reshaped_input = tf.reshape(input, [-1, input_shape[3]])
        softmaxed = tf.nn.softmax(reshaped_input)
        return tf.reshape(
            softmaxed, [-1, input_shape[1], input_shape[2], input_shape[3]]
        )

def spatial_reshape_layer(input, num_dim):
    with tf.name_scope('SpatialReshapeLayer'):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(
            input, [
                input_shape[0],
                input_shape[1],
                -1,
                int(num_dim)
            ]
        )

"""
TODO: From here down: stolen from Tensorflow/models
"""


def expanded_shape(orig_shape, start_dim, num_dims):
    """Inserts multiple ones into a shape vector.

    Inserts an all-1 vector of length num_dims at position start_dim into a shape.
    Can be combined with tf.reshape to generalize tf.expand_dims.

    Args:
    orig_shape: the shape into which the all-1 vector is added (int32 vector)
    start_dim: insertion position (int scalar)
    num_dims: length of the inserted all-1 vector (int scalar)
    Returns:
    An int32 vector of length tf.size(orig_shape) + num_dims.
    """
    with tf.name_scope('ExpandedShape'):
        start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
        before = tf.slice(orig_shape, [0], start_dim)
        add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
        after = tf.slice(orig_shape, start_dim, [-1])
        new_shape = tf.concat([before, add_shape, after], 0)
        return new_shape


def meshgrid(x, y):
    """Tiles the contents of x and y into a pair of grids.

    Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
    are vectors. Generally, this will give:

    xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
    ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)

    Keep in mind that the order of the arguments and outputs is reverse relative
    to the order of the indices they go into, done for compatibility with numpy.
    The output tensors have the same shapes.  Specifically:

    xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
    ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())

    Args:
    x: A tensor of arbitrary shape and rank. xgrid will contain these values
       varying in its last dimensions.
    y: A tensor of arbitrary shape and rank. ygrid will contain these values
       varying in its first dimensions.
    Returns:
    A tuple of tensors (xgrid, ygrid).
    """
    with tf.name_scope('Meshgrid'):
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
        y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

        xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
        ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
        new_shape = y.get_shape().concatenate(x.get_shape())
        xgrid.set_shape(new_shape)
        ygrid.set_shape(new_shape)

    return xgrid, ygrid