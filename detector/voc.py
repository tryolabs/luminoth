import click
import tensorflow as tf
import os

from lxml import etree


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
    with open(path) as f:
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
    with open(path, 'rb') as f:
        image = f.read()
    return image


def load_split(root, split='train'):
    """
    Returns the image identifiers corresponding to the split `split` (values:
    'train', 'val', 'test').
    """
    if split not in ['train', 'val', 'test']:
        raise ValueError

    split_path = os.path.join(root, 'ImageSets', 'Main', f'{split}.txt')
    with open(split_path) as f:
        for line in f:
            yield line.strip()


def get_image_path(root, image_id):
    return os.path.join(root, 'JPEGImages', f'{image_id}.jpg')


def get_image_annotation(root, image_id):
    return os.path.join(root, 'Annotations', f'{image_id}.xml')


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

    object_feature_lists = {
        'label': tf.train.FeatureList(feature=[_int64(classes.index(b['name'])) for b in annotation['object']]),
        'xmin': tf.train.FeatureList(feature=[_int64(b['bndbox']['xmin']) for b in annotation['object']]),
        'ymin': tf.train.FeatureList(feature=[_int64(b['bndbox']['ymin']) for b in annotation['object']]),
        'xmax': tf.train.FeatureList(feature=[_int64(b['bndbox']['xmax']) for b in annotation['object']]),
        'ymax': tf.train.FeatureList(feature=[_int64(b['bndbox']['ymax']) for b in annotation['object']]),
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
    example = tf.train.SequenceExample(feature_lists=object_features, context=context)
    # example = tf.train.SequenceExample(feature_lists=features)

    return example


@click.command()
@click.option('--data-dir', default='datasets/voc')
@click.option('--output-dir', default='datasets/voc/tf')
@click.option('splits', '--split', multiple=True, default=['train', 'val', 'test'])
@click.option('ignore_splits', '--ignore-split', multiple=True)
def voc(data_dir, output_dir, splits, ignore_splits):
    """
    Prepare VOC dataset for ingestion.

    Converts the VOC dataset into three (one per split) TFRecords files.
    """
    print(f'Saving output_dir = {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    classes = read_classes(data_dir)
    splits = [s for s in splits if s not in set(ignore_splits)]
    print(f'Generating outputs for splits = {", ".join(splits)}')

    for split in splits:
        print(f'Converting split = {split}')
        record_file = os.path.join(output_dir, f'{split}.tfrecords')
        writer = tf.python_io.TFRecordWriter(record_file)

        for image_id in load_split(data_dir, split):
            example = image_to_example(data_dir, classes, image_id)
            writer.write(example.SerializeToString())

        writer.close()
