import os
import tensorflow as tf

from luminoth.utils.dataset import (
    read_xml, read_image, to_int64, to_string, to_bytes
)
from .dataset import DatasetTool, InvalidDataDirectory


class PascalVOC(DatasetTool):

    def __init__(self, data_dir):
        super(PascalVOC, self).__init__()
        self._data_dir = data_dir
        self._labels_path = os.path.join(self._data_dir, 'ImageSets', 'Main')
        self._images_path = os.path.join(self._data_dir, 'JPEGImages')
        self._annots_path = os.path.join(self._data_dir, 'Annotations')
        self.is_valid()

    def is_valid(self):
        if not tf.gfile.Exists(self._data_dir):
            raise InvalidDataDirectory(
                '"{}" does not exist.'.format(self._data_dir)
            )

        if not tf.gfile.Exists(self._labels_path):
            raise InvalidDataDirectory('Labels path is missing')

        if not tf.gfile.Exists(self._images_path):
            raise InvalidDataDirectory('Images path is missing')

        if not tf.gfile.Exists(self._annots_path):
            raise InvalidDataDirectory('Annotations path is missing')

    def read_classes(self):
        classes = set()
        for entry in tf.gfile.ListDirectory(self._labels_path):
            if "_" not in entry:
                continue
            class_name, _ = entry.split('_')
            classes.add(class_name)

        return list(sorted(classes))

    def get_split_path(self, split):
        if split not in self.VALID_SPLITS:
            raise ValueError

        split_path = os.path.join(self._labels_path, '{}.txt'.format(split))

        return split_path

    def get_image_path(self, image_id):
        return os.path.join(self._images_path, '{}.jpg'.format(image_id))

    def load_split(self, split='train'):
        split_path = self.get_split_path(split)

        if not tf.gfile.Exists(split_path):
            raise ValueError('"{}" not found.'.format(split_path))

        with tf.gfile.GFile(split_path) as f:
            for line in f:
                yield line.strip()

    def get_split_size(self, split):
        total_records = 0
        for line in self.load_split(split):
            total_records += 1

        return total_records

    def get_image_annotation(self, image_id):
        return os.path.join(self._annots_path, '{}.xml'.format(image_id))

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
