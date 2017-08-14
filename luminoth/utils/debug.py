import tensorflow as tf

# flake8: noqa


def debug(*args, **kwargs):
    def call_ipdb(*args, **kwargs):
        print(args)
        print(kwargs)
        import ipdb; ipdb.set_trace()
        return 0

    return tf.py_func(call_ipdb,
        [list(args) + list(kwargs.values())],
        tf.int32
    )