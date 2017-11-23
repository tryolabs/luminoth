import json
import os
import tensorflow as tf

from luminoth.tools.dataset.readers import InvalidDataDirectory
from luminoth.tools.dataset.readers.object_detection import (
    ObjectDetectionReader
)
from luminoth.utils.dataset import read_image


DEFAULT_YEAR = '2017'


class COCOReader(ObjectDetectionReader):
    def __init__(self, data_dir, split, year=DEFAULT_YEAR,
                 use_supercategory=False, **kwargs):
        super(COCOReader, self).__init__(**kwargs)
        self._data_dir = data_dir
        self._split = split
        self._year = year

        try:
            if self._split == 'train':
                tf.logging.debug('Loading annotation json (may take a while).')

            annotations_json = json.load(
                tf.gfile.Open(self._get_annotations_path())
            )
        except tf.errors.NotFoundError:
            raise InvalidDataDirectory(
                'Could not find COCO annotations in path'
            )

        self._total_records = len(annotations_json['images'])

        category_to_name = {
            c['id']: (c['supercategory'] if use_supercategory else c['name'])
            for c in annotations_json['categories']
        }

        self._total_classes = sorted(set(category_to_name.values()))

        self._image_to_bboxes = {}
        for annotation in annotations_json['annotations']:
            image_id = annotation['image_id']
            x, y, width, height = annotation['bbox']
            self._image_to_bboxes.setdefault(image_id, []).append({
                'xmin': x,
                'ymin': y,
                'xmax': x + width,
                'ymax': y + height,
                'label': self.classes.index(
                    category_to_name[annotation['category_id']]
                ),
            })

        self._image_to_details = {}
        for image in annotations_json['images']:
            self._image_to_details[image['id']] = {
                'file_name': image['file_name'],
                'width': image['width'],
                'height': image['height'],
            }

        del annotations_json

        self.yielded_records = 0
        self.errors = 0

    def get_total(self):
        return self._total_records

    def get_classes(self):
        return self._total_classes

    def iterate(self):
        for image_id, image_details in self._image_to_details.items():

            if self._stop_iteration():
                return

            if not self._is_valid(image_id):
                continue

            filename = image_details['file_name']
            width = image_details['width']
            height = image_details['height']

            try:
                image_path = self._get_image_path(filename)
                image = read_image(image_path)
            except tf.errors.NotFoundError:
                tf.logging.debug(
                    'Error reading image or annotation for "{}".'.format(
                        image_id))
                self.errors += 1
                continue

            gt_boxes = self._image_to_bboxes.get(image_id, [])
            if len(gt_boxes) == 0:
                continue

            self.yielded_records += 1

            yield {
                'width': width,
                'height': height,
                'depth': 3,
                'filename': filename,
                'image_raw': image,
                'gt_boxes': gt_boxes,
            }

    def _get_annotations_path(self):
        filename = 'instances_{}{}.json'.format(self._split, self._year)
        base_dir = os.path.join(self._data_dir, filename)
        if tf.gfile.Exists(base_dir):
            return base_dir

        return os.path.join(self._data_dir, 'annotations', filename)

    def _get_image_path(self, image):
        return os.path.join(
            self._data_dir,
            '{}{}'.format(self._split, self._year),
            image
        )
