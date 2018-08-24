import collections
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

VALID_KEYS = [
    ('x', 'y', 'width', 'height', 'label'),
    ('x_min', 'y_min', 'x_max', 'y_max', 'label')
]


class TaggerineReader(ObjectDetectionReader):
    """
    Object detection reader for data tagged using Taggerine:
    https://github.com/tryolabs/taggerine/
    """
    def __init__(self, data_dir, split, default_class=0, **kwargs):
        super(TaggerineReader, self).__init__(**kwargs)
        self._data_dir = data_dir
        self._split = split
        self._split_path = os.path.join(self._data_dir, self._split)
        self._default_class = default_class

        self.annotations = []
        # Find, read and "parse" annotations from files.
        self._read_annotations(self._split_path)

        self.errors = 0
        self.yielded_records = 0

    def get_total(self):
        """Returns the number of files annotated.
        """
        return len(self.annotations)

    def get_classes(self):
        """Returns the sorted list of possible labels.
        """
        return sorted(set([
            b.get('label', self._default_class)
            for r in self.annotations
            for b in r.get('gt_boxes')
        ]))

    def _read_annotations(self, directory):
        """
        Finds and parses Taggerine's JSON files.
        """
        try:
            all_files = tf.gfile.ListDirectory(self._split_path)
        except tf.errors.NotFoundError:
            raise InvalidDataDirectory(
                'Directory for split "{}" does not exist'.format(
                    self._split))

        annotation_file_candidates = []
        for filename in all_files:
            if filename.lower().endswith('.json'):
                annotation_file_candidates.append(filename)

        if len(annotation_file_candidates) == 0:
            raise InvalidDataDirectory(
                'Could not find any annotations in {}.'.format(
                    self._split_path) +
                'Check that there is a .json file with Taggerine\'s ' +
                'annotations.')

        self.annotations = []
        # Open, validate and extract label information.
        for filename in annotation_file_candidates:
            annotation_path = os.path.join(self._split_path, filename)
            with tf.gfile.Open(annotation_path) as annotation_file:
                annotations = json.load(annotation_file)

            if not isinstance(annotations, dict):
                # JSON file with invalid format.
                continue

            file_annotations = []

            invalid_label = False
            for image_filename, labels in annotations.items():
                if not isinstance(labels, collections.Iterable):
                    # Ignore labels that are not lists. Ignore file.
                    invalid_label = True
                    break

                # Validate labels
                for label in labels:
                    for valid_keyset in VALID_KEYS:
                        if all(key in label for key in valid_keyset):
                            break
                    else:
                        # There is not valid_keyset that can parse the label.
                        # Ignore all labels from this file.
                        invalid_label = True
                        break

                # Early stop for labels inside file when there is an invalid
                # label.
                if invalid_label:
                    break

                # Save annotations.
                file_annotations.append({
                    'image_id': os.path.basename(image_filename),
                    'filename': image_filename,
                    'path': os.path.join(self._split_path, image_filename),
                    'gt_boxes': labels,
                })

            if invalid_label:
                # Ignore file that have invalid labels.
                continue

            # If we have a valid file with data in it then we use it.
            self.annotations.extend(file_annotations)

    def iterate(self):
        for annotation in self.annotations:
            # Checks that we don't yield more records than necessary.
            if self._stop_iteration():
                return

            image_id = annotation['image_id']

            if self._should_skip(image_id):
                continue

            try:
                image = read_image(annotation['path'])
            except tf.errors.NotFoundError:
                tf.logging.debug(
                    'Error reading image or annotation for "{}".'.format(
                        image_id))
                self.errors += 1
                continue

            # Parse image bytes with PIL to get width and height.
            image_pil = Image.open(six.BytesIO(image))
            img_width = image_pil.width
            img_height = image_pil.height

            gt_boxes = []
            for b in annotation['gt_boxes']:
                try:
                    label_id = self.classes.index(
                        b.get('label', self._default_class)
                    )
                except ValueError:
                    continue

                if 'height' in b and 'width' in b and 'x' in b and 'y' in b:
                    gt_boxes.append({
                        'label': label_id,
                        'xmin': b['x'] * img_width,
                        'ymin': b['y'] * img_height,
                        'xmax': b['x'] * img_width + b['width'] * img_width,
                        'ymax': b['y'] * img_height + b['height'] * img_height,
                    })
                else:
                    gt_boxes.append({
                        'label': label_id,
                        'xmin': b['x_min'] * img_width,
                        'ymin': b['y_min'] * img_height,
                        'xmax': b['x_max'] * img_width,
                        'ymax': b['y_max'] * img_height,
                    })

            if len(gt_boxes) == 0:
                tf.logging.debug('Image "{}" has zero valid gt_boxes.'.format(
                    image_id))
                self.errors += 1
                continue

            record = {
                'width': img_width,
                'height': img_height,
                'depth': 3,
                'filename': image_id,
                'image_raw': image,
                'gt_boxes': gt_boxes,
            }

            self._will_add_record(record)
            self.yielded_records += 1

            yield record
