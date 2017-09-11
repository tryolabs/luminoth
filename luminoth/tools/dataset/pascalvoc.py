import click
import os
import tensorflow as tf

from luminoth.utils.dataset import (
    read_xml, read_image, to_int64, to_string, to_bytes
)
from .dataset import DatasetTool, RecordSaver


class PascalVOC(DatasetTool):

    def __init__(self, data_dir):
        super(PascalVOC, self).__init__()
        self._data_dir = data_dir

    def read_classes(self):
        path = os.path.join(self._data_dir, 'ImageSets', 'Main')

        classes = set()
        for entry in tf.gfile.ListDirectory(path):
            if "_" not in entry:
                continue
            class_name, _ = entry.split('_')
            classes.add(class_name)

        return list(sorted(classes))

    def load_split(self, split='train'):
        if split not in self.VALID_SPLITS:
            raise ValueError

        split_path = os.path.join(
            self._data_dir, 'ImageSets', 'Main', '{}.txt'.format(split))
        with tf.gfile.GFile(split_path) as f:
            for line in f:
                yield line.strip()

    def get_image_path(self, image_id):
        return os.path.join(
            self._data_dir, 'JPEGImages', '{}.jpg'.format(image_id)
        )

    def get_image_annotation(self, image_id):
        return os.path.join(
            self._data_dir, 'Annotations', '{}.xml'.format(image_id)
        )

    def image_to_example(self, classes, image_id):
        annotation_path = self.get_image_annotation(image_id)
        image_path = self.get_image_path(image_id)

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

            object_features_values['label'].append(
                to_int64(label_id)
            )
            object_features_values['xmin'].append(
                to_int64(b['bndbox']['xmin'])
            )
            object_features_values['ymin'].append(
                to_int64(b['bndbox']['ymin'])
            )
            object_features_values['xmax'].append(
                to_int64(b['bndbox']['xmax'])
            )
            object_features_values['ymax'].append(
                to_int64(b['bndbox']['ymax'])
            )

        if len(object_features_values['label']) == 0:
            # No bounding box matches the available classes.
            return

        object_feature_lists = {
            'label': tf.train.FeatureList(
                feature=object_features_values['label']
            ),
            'xmin': tf.train.FeatureList(
                feature=object_features_values['xmin']
            ),
            'ymin': tf.train.FeatureList(
                feature=object_features_values['ymin']
            ),
            'xmax': tf.train.FeatureList(
                feature=object_features_values['xmax']
            ),
            'ymax': tf.train.FeatureList(
                feature=object_features_values['ymax']
            ),
        }

        object_features = tf.train.FeatureLists(
            feature_list=object_feature_lists
        )

        sample = {
            'width': to_int64(int(annotation['size']['width'])),
            'height': to_int64(int(annotation['size']['height'])),
            'depth': to_int64(int(annotation['size']['depth'])),
            'filename': to_string(annotation['filename']),
            'image_raw': to_bytes(image),
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
@click.option('ignore_splits', '--ignore-split', multiple=True)
@click.option('--only-filename', help='Create dataset with a single example.')
@click.option('--limit-examples', type=int, help='Limit dataset with to the first `N` examples.')  # noqa
@click.option('--limit-classes', type=int, help='Limit dataset with `N` random classes.')  # noqa
@click.option('--seed', type=int, help='Seed used for picking random classes.')
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def pascalvoc(data_dir, output_dir, ignore_splits, only_filename,
              limit_examples, limit_classes, seed, debug):
    """
    Prepares VOC dataset for ingestion.

    Converts the VOC dataset into three (one per split) TFRecords files.
    """
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    voc = PascalVOC(data_dir=data_dir)
    saver = RecordSaver(
        voc, output_dir,
        ignore_splits=ignore_splits,
        only_filename=only_filename,
        limit_examples=limit_examples,
        limit_classes=limit_classes,
        seed=seed
    )

    saver.save()
