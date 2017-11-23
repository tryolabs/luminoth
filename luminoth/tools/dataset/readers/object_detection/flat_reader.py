import json
import os
import six
import tensorflow as tf

from PIL import Image

from luminoth.tools.dataset.readers import InvalidDataDirectory
from luminoth.tools.dataset.readers.object_detection import (
    ObjectDetectionReader
)
from luminoth.utils.dataset import read_image

DEFAULT_ANNOTATION_TYPE = 'json'
DEFAULT_CLASS = 0
DEFAULT_OBJECTS_KEY = 'rects'
X_MIN_KEY = 'x1'
Y_MIN_KEY = 'y1'
X_MAX_KEY = 'x2'
Y_MAX_KEY = 'y2'


class FlatReader(ObjectDetectionReader):
    def __init__(self, data_dir, split,
                 annotation_type=DEFAULT_ANNOTATION_TYPE,
                 default_class=DEFAULT_CLASS, objects_key=DEFAULT_OBJECTS_KEY,
                 x_min_key=X_MIN_KEY, y_min_key=Y_MIN_KEY, x_max_key=X_MAX_KEY,
                 y_max_key=Y_MAX_KEY, **kwargs):
        super(FlatReader, self).__init__(**kwargs)
        self._data_dir = data_dir
        self._split = split
        self._annotation_type = annotation_type
        self._default_class = default_class
        self._objects_key = objects_key
        self._x_min_key = x_min_key
        self._y_min_key = y_min_key
        self._x_max_key = x_max_key
        self._y_max_key = y_max_key

        self._annotated_files = None
        self._annotations = None

        self.errors = 0
        self.yielded_records = 0

    def get_total(self):
        return len(self.annotated_files)

    def get_classes(self):
        return sorted(set([
            b.get('label', self._default_class)
            for r in self.annotations
            for b in r.get(self._objects_key, [])
        ]))

    @property
    def annotated_files(self):
        if self._annotated_files is None:
            split_path = self._get_split_path()
            try:
                all_files = tf.gfile.ListDirectory(split_path)
            except tf.errors.NotFoundError:
                raise InvalidDataDirectory(
                    'Directory for split "{}" does not exist'.format(
                        self._split))

            self._annotated_files = []
            for filename in all_files:
                if filename.endswith('.{}'.format(self._annotation_type)):
                    self._annotated_files.append(
                        filename[:-(len(self._annotation_type) + 1)]
                    )
            if len(self._annotated_files) == 0:
                raise InvalidDataDirectory(
                    'Could not find any annotations in {}'.format(split_path))

        return self._annotated_files

    def iterate(self):
        for annotation in self.annotations:
            if self._stop_iteration():
                return

            image_id = annotation['image_id']

            if not self._is_valid(image_id):
                continue

            try:
                image_path = self._get_image_path(image_id)
                image = read_image(image_path)
            except tf.errors.NotFoundError:
                tf.logging.debug(
                    'Error reading image or annotation for "{}".'.format(
                        image_id))
                self.errors += 1
                continue

            image_pil = Image.open(six.BytesIO(image))
            width = image_pil.width
            height = image_pil.height

            gt_boxes = []
            for b in annotation[self._objects_key]:
                try:
                    label_id = self.classes.index(
                        b.get('label', self._default_class)
                    )
                except ValueError:
                    continue

                gt_boxes.append({
                    'label': label_id,
                    'xmin': b[self._x_min_key],
                    'ymin': b[self._y_min_key],
                    'xmax': b[self._x_max_key],
                    'ymax': b[self._y_max_key],
                })

            if len(gt_boxes) == 0:
                tf.logging.debug('Image "{}" has zero valid gt_boxes.'.format(
                    image_id))
                self.errors += 1
                continue

            self.yielded_records += 1

            yield {
                'width': width,
                'height': height,
                'depth': 3,
                'filename': image_id,
                'image_raw': image,
                'gt_boxes': gt_boxes,
            }

    @property
    def annotations(self):
        if self._annotations is None:
            self._annotations = []
            for annotation_id in self.annotated_files:
                annotation = self._read_annotation(
                    self._get_annotation_path(annotation_id)
                )
                annotation['image_id'] = annotation_id
                self._annotations.append(annotation)
        return self._annotations

    def _get_annotation_path(self, annotation_id):
        return os.path.join(
            self._get_split_path(),
            '{}.{}'.format(annotation_id, self._annotation_type)
        )

    def _get_split_path(self):
        return os.path.join(self._data_dir, self._split)

    def _get_image_path(self, image_id):
        default_path = os.path.join(self._get_split_path(), image_id)
        if tf.gfile.Exists(default_path):
            return default_path

        # Lets assume it doesn't have extension
        possible_files = [
            f for f in self.files if f.startswith('{}.'.format(image_id))
        ]
        if len(possible_files) == 0:
            return
        elif len(possible_files) > 1:
            tf.logging.warning(
                'Image {} matches with {} files ({}).'.format(
                    image_id, len(possible_files)))

        return os.path.join(self._get_split_path(), possible_files[0])

    def _read_annotation(self, annotation_path):
        if self._annotation_type == 'json':
            with tf.gfile.Open(annotation_path) as annotation_file:
                return json.load(annotation_file)
        else:
            raise ValueError(
                'Annotation type {} not supported'.format(
                    self._annotation_type))
