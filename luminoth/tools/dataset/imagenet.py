import click
import json
import os
import random
import tensorflow as tf

from PIL import Image

from luminoth.utils.dataset import (
    read_xml, read_image, _int64, _string, _bytes
)


DEFAULT_TOTAL_CLASSES = 20


def adjust_bbox(xmin, ymin, xmax, ymax, old_width, old_height,
                new_width, new_height):
    # TODO: consider reusing luminoth.utils.image.adjust_bboxes instead of
    # this, but note it uses tensorflow, and using tf and np here may introduce
    # too many problems.
    xmin = (xmin / old_width) * new_width
    ymin = (ymin / old_height) * new_height
    xmax = (xmax / old_width) * new_width
    ymax = (ymax / old_height) * new_height

    return xmin, ymin, xmax, ymax


def read_classes(root):
    # TODO: find a more robust way without doing something as wasteful as
    # parsing all xml annotations to get the objects mentioned.
    # ILSVRC2014 added no new catgories, so this works.
    path = os.path.join(
        root, 'Annotations', 'DET', 'train', 'ILSVRC2013_train'
    )
    classes = set()
    for entry in tf.gfile.ListDirectory(path):
        classes.add(entry)

    return list(sorted(classes))


def load_split(root, split='train'):
    if split not in ['train', 'val', 'test']:
        raise ValueError

    split_path = os.path.join(
        root, 'ImageSets', 'DET', '{}.txt'.format(split)
    )
    with tf.gfile.GFile(split_path) as f:
        for line in f:
            # The images in 'extra' directories don't have annotations.
            if 'extra' in line:
                continue
            filename = line.split()[0]
            filename = os.path.join(split, filename)
            yield filename.strip()


def get_image_path(data_dir, image_id):
    return os.path.join(data_dir, 'Data', 'DET', '{}.JPEG'.format(image_id))


def get_image_annotation(data_dir, image_id):
    return os.path.join(
        data_dir, 'Annotations', 'DET', '{}.xml'.format(image_id)
    )


def image_to_example(data_dir, classes, image_id):
    annotation_path = get_image_annotation(data_dir, image_id)
    image_path = get_image_path(data_dir, image_id)

    # Read both the image and the annotation into memory.
    annotation = read_xml(annotation_path)
    image = read_image(image_path)

    # TODO: consider alternatives to using Pillow here.
    image_pil = Image.open(image_path)
    width = image_pil.width
    height = image_pil.height
    image_pil.close()

    object_features_values = {
        'label': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
    }

    objects = annotation.get('object')
    if objects is None:
        # If there's no bounding boxes, we don't want it
        return
    for b in annotation['object']:
        try:
            label_id = classes.index(b['name'])
        except ValueError:
            continue

        (xmin, ymin, xmax, ymax) = adjust_bbox(
            xmin=int(b['bndbox']['xmin']), ymin=int(b['bndbox']['ymin']),
            xmax=int(b['bndbox']['xmax']), ymax=int(b['bndbox']['ymax']),
            old_width=int(annotation['size']['width']),
            old_height=int(annotation['size']['height']),
            new_width=width, new_height=height
        )
        object_features_values['label'].append(_int64(label_id))
        object_features_values['xmin'].append(_int64(xmin))
        object_features_values['ymin'].append(_int64(ymin))
        object_features_values['xmax'].append(_int64(xmax))
        object_features_values['ymax'].append(_int64(ymax))

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
        'width': _int64(width),
        'height': _int64(height),
        'depth': _int64(3),
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
def imagenet(data_dir, output_dir, splits, ignore_splits, only_filename,
             limit_examples, limit_classes, seed, debug):
    """
    Prepares ImageNet dataset for ingestion.
    """
    # TODO: this is a copy-paste from voc.py
    # It is pretty close to working as it is, but consider rewriting this or
    # reusing (without copy-pasting) the voc code.
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Saving output_dir = {}'.format(output_dir))
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

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
