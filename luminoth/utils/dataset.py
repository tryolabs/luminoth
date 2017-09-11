import tensorflow as tf

from lxml import etree


def node2dict(root):
    if root.getchildren():
        val = {}
        for node in root.getchildren():
            chkey, chval = node2dict(node)
            val[chkey] = chval
    else:
        val = root.text

    return root.tag, val


def read_xml(path):
    with tf.gfile.GFile(path) as f:
        root = etree.fromstring(f.read())

    annotations = {}
    for node in root.getchildren():
        key, val = node2dict(node)
        # If `key` is object, it's actually a list.
        if key == 'object':
            annotations.setdefault(key, []).append(val)
        else:
            annotations[key] = val

    return annotations


def read_image(path):
    with tf.gfile.GFile(path, 'rb') as f:
        image = f.read()
    return image


def to_int64(value):
    value = [int(value)] if not isinstance(value, list) else value
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)
    )


def to_bytes(value):
    value = [value] if not isinstance(value, list) else value
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )


def to_string(value):
    value = [value] if not isinstance(value, list) else value
    value = [v.encode('utf-8') for v in value]
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )
