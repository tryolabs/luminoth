import json
import os
import six
import tensorflow as tf

from PIL import Image

from luminoth.tools.dataset.readers import InvalidDataDirectory
from luminoth.tools.dataset.readers.object_detection import (
    ObjectDetectionReader
)
from luminoth.utils.dataset import read_xml, read_image

WNIDS_FILE = 'data/imagenet_wnids.json'


class ImageNetReader(ObjectDetectionReader):
    def __init__(self, data_dir, split, **kwargs):
        super(ImageNetReader, self).__init__(**kwargs)
        self._split = split
        self._data_dir = data_dir
        self._imagesets_path = os.path.join(self._data_dir, 'ImageSets', 'DET')
        self._images_path = os.path.join(self._data_dir, 'Data', 'DET',)
        self._annotations_path = os.path.join(
            self._data_dir, 'Annotations', 'DET'
        )

        self.yielded_records = 0
        self.errors = 0

        # Validate Imagenet structure in `data_dir`.
        self._validate_structure()

        # Load wnids from file.
        wnids_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            WNIDS_FILE
        )
        with tf.gfile.GFile(wnids_path) as wnidsjson:
            self._wnids = json.load(wnidsjson)

    def get_total(self):
        return sum(1 for _ in self._get_record_names())

    def get_classes(self):
        return sorted(list(self._wnids.values()))

    def iterate(self):
        for image_id in self._get_record_names():
            if self._stop_iteration():
                return

            if not self._is_valid(image_id):
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

            objects = annotation.get('object')
            if objects is None:
                # If there's no bounding boxes, we don't want it.
                continue

            image_pil = Image.open(six.BytesIO(image))
            width = image_pil.width
            height = image_pil.height

            gt_boxes = []
            for b in annotation['object']:
                try:
                    label_id = self.classes.index(self._wnids[b['name']])
                except ValueError:
                    continue

                (xmin, ymin, xmax, ymax) = self._adjust_bbox(
                    xmin=int(b['bndbox']['xmin']),
                    ymin=int(b['bndbox']['ymin']),
                    xmax=int(b['bndbox']['xmax']),
                    ymax=int(b['bndbox']['ymax']),
                    old_width=int(annotation['size']['width']),
                    old_height=int(annotation['size']['height']),
                    new_width=width, new_height=height
                )

                gt_boxes.append({
                    'label': label_id,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                })

            if len(gt_boxes) == 0:
                continue

            self.yielded_records += 1

            yield {
                'width': width,
                'height': height,
                'depth': 3,
                'filename': annotation['filename'],
                'image_raw': image,
                'gt_boxes': gt_boxes,
            }

    def _validate_structure(self):
        if not tf.gfile.Exists(self._data_dir):
            raise InvalidDataDirectory(
                '"{}" does not exist.'.format(self._data_dir)
            )

        if not tf.gfile.Exists(self._imagesets_path):
            raise InvalidDataDirectory('ImageSets path is missing')

        if not tf.gfile.Exists(self._images_path):
            raise InvalidDataDirectory('Images path is missing')

        if not tf.gfile.Exists(self._annotations_path):
            raise InvalidDataDirectory('Annotations path is missing')

    def _get_split_path(self):
        return os.path.join(
            self._imagesets_path, '{}.txt'.format(self._split)
        )

    def _get_image_path(self, image_id):
        return os.path.join(
            self._images_path, '{}.JPEG'.format(image_id)
        )

    def _get_image_annotation(self, image_id):
        return os.path.join(self._annotations_path, '{}.xml'.format(image_id))

    def _get_record_names(self):
        split_path = self._get_split_path()

        if not tf.gfile.Exists(split_path):
            raise ValueError('"{}" not found'.format(self._split))

        with tf.gfile.GFile(split_path) as f:
            for line in f:
                # The images in 'extra' directories don't have annotations.
                if 'extra' in line:
                    continue
                filename = line.split()[0]
                filename = os.path.join(self._split, filename)
                yield filename.strip()

    def _adjust_bbox(self, xmin, ymin, xmax, ymax, old_width, old_height,
                     new_width, new_height):
        # TODO: consider reusing luminoth.utils.image.adjust_bboxes instead of
        # this, but note it uses tensorflow, and using tf and np here may
        # introduce too many problems.
        xmin = (xmin / old_width) * new_width
        ymin = (ymin / old_height) * new_height
        xmax = (xmax / old_width) * new_width
        ymax = (ymax / old_height) * new_height

        return xmin, ymin, xmax, ymax
