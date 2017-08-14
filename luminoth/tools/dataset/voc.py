import click
import json
import os
import random
import tensorflow as tf

from lxml import etree


DEFAULT_TOTAL_CLASSES = 20


def read_classes(root):
    path = os.path.join(root, 'ImageSets', 'Main')

    classes = set()
    for entry in os.listdir(path):
        if "_" not in entry:
            continue
        class_name, _ = entry.split('_')
        classes.add(class_name)

    return list(sorted(classes))


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


def load_split(root, split='train'):
    """
    Returns the image identifiers corresponding to the split `split` (values:
    'train', 'val', 'test').
    """
    if split not in ['train', 'val', 'test']:
        raise ValueError

    split_path = os.path.join(
        root, 'ImageSets', 'Main', '{}.txt'.format(split))
    with tf.gfile.GFile(split_path) as f:
        for line in f:
            yield line.strip()


def get_image_path(root, image_id):
    return os.path.join(root, 'JPEGImages', '{}.jpg'.format(image_id))


def get_image_annotation(root, image_id):
    return os.path.join(root, 'Annotations', '{}.xml'.format(image_id))


def _int64(value):
    value = [int(value)] if not isinstance(value, list) else value
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value)
    )


def _bytes(value):
    value = [value] if not isinstance(value, list) else value
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )


def _string(value):
    value = [value] if not isinstance(value, list) else value
    value = [v.encode('utf-8') for v in value]
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value)
    )


def image_to_example(data_dir, classes, image_id):
    annotation_path = get_image_annotation(data_dir, image_id)
    image_path = get_image_path(data_dir, image_id)

    # Read both the image and the annotation into memory.
    annotation = read_xml(annotation_path)
    image = read_image(image_path)

    object_features_values = {
        'label': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
    }

    for b in annotation['object']:
        try:
            label_id = classes.index(b['name'])
        except ValueError:
            continue

        object_features_values['label'].append(_int64(label_id))
        object_features_values['xmin'].append(_int64(b['bndbox']['xmin']))
        object_features_values['ymin'].append(_int64(b['bndbox']['ymin']))
        object_features_values['xmax'].append(_int64(b['bndbox']['xmax']))
        object_features_values['ymax'].append(_int64(b['bndbox']['ymax']))

    if len(object_features_values['label']) == 0:
        # No bounding box matches the available classes.
        return

    object_feature_lists = {
        'label': tf.train.FeatureList(feature=object_features_values['label']),
        'xmin': tf.train.FeatureList(feature=object_features_values['xmin']),
        'ymin': tf.train.FeatureList(feature=object_features_values['ymin']),
        'xmax': tf.train.FeatureList(feature=object_features_values['xmax']),
        'ymax': tf.train.FeatureList(feature=object_features_values['ymax']),
    }

    object_features = tf.train.FeatureLists(feature_list=object_feature_lists)

    sample = {
        'width': _int64(int(annotation['size']['width'])),
        'height': _int64(int(annotation['size']['height'])),
        'depth': _int64(int(annotation['size']['depth'])),
        'filename': _string(annotation['filename']),
        'image_raw': _bytes(image),
    }

    # Now build an `Example` protobuf object and save with the writer.
    context = tf.train.Features(feature=sample)
    example = tf.train.SequenceExample(
        feature_lists=object_features, context=context
    )

    return example


@click.command()
@click.option('--data-dir', default='datasets/voc')
@click.option('--output-dir', default='datasets/voc/tf')
@click.option('splits', '--split', multiple=True, default=['train', 'val', 'test'])  # noqa
@click.option('ignore_splits', '--ignore-split', multiple=True)
@click.option('--only-filename', help='Create dataset with a single example.')
@click.option('--limit-examples', type=int, help='Limit dataset with to the first `N` examples.')  # noqa
@click.option('--limit-classes', type=int, default=DEFAULT_TOTAL_CLASSES, help='Limit dataset with `N` random classes.')  # noqa
@click.option('--seed', type=int, default=0, help='Seed used for picking random classes.')  # noqa
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def voc(data_dir, output_dir, splits, ignore_splits, only_filename,
        limit_examples, limit_classes, seed, debug):
    """
    Prepare VOC dataset for ingestion.

    Converts the VOC dataset into three (one per split) TFRecords files.
    """
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Saving output_dir = {}'.format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    classes = read_classes(data_dir)

    if limit_classes < DEFAULT_TOTAL_CLASSES:
        random.seed(seed)
        classes = random.sample(classes, limit_classes)
        tf.logging.info('Limiting to {} classes: {}'.format(
            limit_classes, classes
        ))

    if only_filename:
        classes_filename = 'classes-{}.json'.format(only_filename)
    elif limit_examples:
        classes_filename = 'classes-top{}-{}classes.json'.format(
            limit_examples, limit_classes
        )
    else:
        classes_filename = 'classes.json'

    classes_file = os.path.join(output_dir, classes_filename)

    json.dump(classes, tf.gfile.GFile(classes_file, 'w'))

    splits = [s for s in splits if s not in set(ignore_splits)]
    tf.logging.debug(
        'Generating outputs for splits = {}'.format(", ".join(splits)))

    for split in splits:
        tf.logging.debug('Converting split = {}'.format(split))
        if only_filename:
            record_filename = '{}-{}.tfrecords'.format(split, only_filename)
        elif limit_examples:
            record_filename = '{}-top{}-{}classes.tfrecords'.format(
                split, limit_examples, limit_classes
            )
        else:
            record_filename = '{}.tfrecords'.format(split)

        record_file = os.path.join(output_dir, record_filename)
        writer = tf.python_io.TFRecordWriter(record_file)

        total_examples = 0
        for num, image_id in enumerate(load_split(data_dir, split)):
            if not only_filename or only_filename == image_id:
                # Using limit on classes it's possible for an image_to_example
                # to return None (because no classes match).
                example = image_to_example(data_dir, classes, image_id)
                if example:
                    total_examples += 1
                    writer.write(example.SerializeToString())

            if limit_examples and total_examples == limit_examples:
                break

        writer.close()
        tf.logging.info('Saved split {} to "{}"'.format(split, record_file))
