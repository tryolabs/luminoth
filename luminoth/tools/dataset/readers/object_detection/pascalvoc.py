import os
import random
import tensorflow as tf

from luminoth.tools.dataset.readers import InvalidDataDirectory
from luminoth.tools.dataset.readers.object_detection import (
    ObjectDetectionReader
)
from luminoth.utils.dataset import read_xml, read_image


class PascalVOCReader(ObjectDetectionReader):
    def __init__(self, data_dir, split, only_filename=None,
                 limit_examples=None, limit_classes=None, seed=None, **kwargs):
        super(PascalVOCReader, self).__init__()
        self._data_dir = data_dir
        self._split = split
        self._labels_path = os.path.join(self._data_dir, 'ImageSets', 'Main')
        self._images_path = os.path.join(self._data_dir, 'JPEGImages')
        self._annots_path = os.path.join(self._data_dir, 'Annotations')

        self._only_filename = only_filename
        self._limit_examples = limit_examples
        self._limit_classes = limit_classes
        self._seed = seed
        random.seed(seed)

        self._classes = None
        self._total = None

        self.yielded_records = 0
        self.errors = 0

        # Validate PascalVoc structure in `data_dir`.
        self._validate_structure()

    @property
    def total(self):
        if self._total is None:
            # Unfortunately the most efficient way to have the total number of
            # records is to count all the lines the split file.
            total_records = sum(1 for _ in self._get_record_names())

            # Define smaller number of records when limiting examples.
            if self._only_filename is not None:
                self._total = 1
            elif self._limit_examples is not None and self._limit_examples > 0:
                self._total = min(self._limit_examples, total_records)
            else:
                self._total = total_records

        return self._total

    @property
    def classes(self):
        if self._classes is None:
            classes_set = set()
            for entry in tf.gfile.ListDirectory(self._labels_path):
                if "_" not in entry:
                    continue
                class_name, _ = entry.split('_')
                classes_set.add(class_name)
            self._classes = list(sorted(classes_set))

            # Choose random classes when limiting them
            if self._limit_classes is not None and self._limit_classes > 0:
                total_classes = min(len(self._classes), self._limit_classes)
                self._classes = sorted(
                    random.sample(self._classes, total_classes)
                )

        return self._classes

    def _validate_structure(self):
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

    def _get_split_path(self):
        return os.path.join(self._labels_path, '{}.txt'.format(self._split))

    def _get_record_names(self):
        split_path = self._get_split_path()

        if not tf.gfile.Exists(split_path):
            raise ValueError('"{}" not found.'.format(split_path))

        with tf.gfile.GFile(split_path) as f:
            for line in f:
                yield line.strip()

    def _get_image_annotation(self, image_id):
        return os.path.join(self._annots_path, '{}.xml'.format(image_id))

    def _get_image_path(self, image_id):
        return os.path.join(self._images_path, '{}.jpg'.format(image_id))

    def iterate(self):
        for image_id in self._get_record_names():
            if self.yielded_records == self.total:
                # Finish iteration based on predefined total.
                return

            if self._only_filename and image_id != self._only_filename:
                # Ignore image when using `only_filename` and it doesn't match
                continue

            try:
                annotation_path = self._get_image_annotation(image_id)
                image_path = self._get_image_path(image_id)

                # Read both the image and the annotation into memory.
                annotation = read_xml(annotation_path)
                image = read_image(image_path)
            except tf.errors.NotFoundError:
                tf.logging.debug(
                    'Error reading image or annotation for "{}".'.format(
                        image_id))
                self.errors += 1
                continue

            gt_boxes = []

            for b in annotation['object']:
                try:
                    label_id = self.classes.index(b['name'])
                except ValueError:
                    continue

                gt_boxes.append({
                    'label': label_id,
                    'xmin': b['bndbox']['xmin'],
                    'ymin': b['bndbox']['ymin'],
                    'xmax': b['bndbox']['xmax'],
                    'ymax': b['bndbox']['ymax'],
                })

            if len(gt_boxes) == 0:
                continue

            self.yielded_records += 1

            yield {
                'width': annotation['size']['width'],
                'height': annotation['size']['height'],
                'depth': annotation['size']['depth'],
                'filename': annotation['filename'],
                'image_raw': image,
                'gt_boxes': gt_boxes,
            }
