# -*- coding: utf-8 -*-
import csv
import os
import six
import tensorflow as tf

from PIL import Image

from luminoth.tools.dataset.readers import InvalidDataDirectory
from luminoth.tools.dataset.readers.object_detection import (
    ObjectDetectionReader
)
from luminoth.utils.config import is_basestring
from luminoth.utils.dataset import read_image


class CSVReader(ObjectDetectionReader):
    """CSVReader supports reading annotations out of a CSV file.

    The reader requires the following directory structure within `data_dir`:
    * Annotation data (bounding boxes) per split, under the name `{split}.csv`
      on the root directory.
    * Dataset images per split, under the `{split}/` directory on the root
      directory.

    Thus, a CSV dataset directory structure may look as follows::

        .
        ├── train
        │   ├── image_1.jpg
        │   ├── image_2.jpg
        │   └── image_3.jpg
        ├── val
        │   ├── image_4.jpg
        │   ├── image_5.jpg
        │   └── image_6.jpg
        ├── train.csv
        └── val.csv

    The CSV file itself must have the following format::

        image_id,xmin,ymin,xmax,ymax,label
        image_1.jpg,26,594,86,617,cat
        image_1.jpg,599,528,612,541,car
        image_2.jpg,393,477,430,552,dog

    You can skip the header by overriding the `headers` parameter, in which
    case the `columns` option will be used (specified as either a string or a
    comma-separated list of fields). If this is done, the above six columns
    *must* be present. Extra columns will be ignored.
    """

    DEFAULT_COLUMNS = ['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'label']

    def __init__(self, data_dir, split, headers=True, columns=None, **kwargs):
        """Initializes the reader, allowing to override internal settings.

        Arguments:
            data_dir: Path to base directory where all the files are
                located. See class docstring for a description on the expected
                structure.
            split: Split to read. Possible values depend on the dataset itself.
            headers (boolean): Whether the CSV file has headers indicating
                field names, in which case those will be considered.
            columns (list or str): Column names for when `headers` is `False`
                (i.e. the CSV file has no headers). Will be ignored if
                `headers` is `True`.
        """
        super(CSVReader, self).__init__(**kwargs)

        self._data_dir = data_dir
        self._split = split

        self._annotations_path = os.path.join(
            self._data_dir, '{}.csv'.format(self._split)
        )
        if not tf.gfile.Exists(self._annotations_path):
            raise InvalidDataDirectory(
                'CSV annotation file not found. Should be located at '
                '`{}`'.format(self._annotations_path)
            )

        self._images_dir = os.path.join(self._data_dir, self._split)
        if not tf.gfile.Exists(self._images_dir):
            raise InvalidDataDirectory(
                'Image directory not found. Should be located at '
                '`{}`'.format(self._images_dir)
            )

        if columns is not None:
            if is_basestring(columns):
                columns = columns.split(',')
        else:
            columns = self.DEFAULT_COLUMNS
        self._columns = columns
        self._column_names = set(self._columns)

        self._has_headers = headers

        # Cache for the records.
        # TODO: Don't read it all upfront.
        self._records = None

        # Whether the structure of the CSV file has been checked already.
        self._csv_checked = False

        self.errors = 0
        self.yielded_records = 0

    def get_total(self):
        return len(self._get_records())

    def get_classes(self):
        return sorted(set([
            a['label']
            for annotations in self._get_records().values()
            for a in annotations
        ]))

    def iterate(self):
        records = self._get_records()
        for image_id, annotations in records.items():
            if self._stop_iteration():
                return

            if self._should_skip(image_id):
                continue

            image_path = os.path.join(self._images_dir, image_id)
            try:
                image = read_image(image_path)
            except tf.errors.NotFoundError:
                tf.logging.warning(
                    'Image `{}` at `{}` couldn\'t be opened.'.format(
                        image_id, image_path
                    )
                )
                self.errors += 1
                continue

            image_pil = Image.open(six.BytesIO(image))
            width = image_pil.width
            height = image_pil.height

            gt_boxes = []
            for annotation in annotations:
                try:
                    label_id = self.classes.index(annotation['label'])
                except ValueError:
                    tf.logging.warning(
                        'Error finding id for image `{}`, label `{}`.'.format(
                            image_id, annotation['label']
                        )
                    )
                    continue

                gt_boxes.append({
                    'label': label_id,
                    'xmin': annotation['xmin'],
                    'ymin': annotation['ymin'],
                    'xmax': annotation['xmax'],
                    'ymax': annotation['ymax'],
                })

            if len(gt_boxes) == 0:
                continue

            record = {
                'width': width,
                'height': height,
                'depth': 3,
                'filename': image_id,
                'image_raw': image,
                'gt_boxes': gt_boxes,
            }
            self._will_add_record(record)
            self.yielded_records += 1

            yield record

    def _get_records(self):
        """Read all the records out of the CSV file.

        If they've been previously read, just return the records.

        Returns:
            Dictionary mapping `image_id`s to a list of annotations.
        """
        if self._records is None:
            with tf.gfile.Open(self._annotations_path) as annotations:
                if self._has_headers:
                    reader = csv.DictReader(annotations)
                else:
                    # If file has no headers, pass the field names to the CSV
                    # reader.
                    reader = csv.DictReader(
                        annotations, fieldnames=self._columns
                    )

            images_gt_boxes = {}

            for row in reader:
                # When reading the first row, make sure the CSV is correct.
                self._check_csv(row)

                # Then proceed as normal, reading each row and aggregating
                # bounding boxes by image.
                label = self._normalize_row(row)
                image_id = label.pop('image_id')
                images_gt_boxes.setdefault(image_id, []).append(label)

            self._records = images_gt_boxes

        return self._records

    def _check_csv(self, row):
        """Checks whether the CSV has all the necessary columns.

        The actual check is done on the first row only, once the CSV has been
        finally opened and read.
        """
        if not self._csv_checked:
            missing_keys = self._column_names - set(row.keys())
            if missing_keys:
                raise InvalidDataDirectory(
                    'Columns missing from CSV: {}'.format(missing_keys)
                )
            self._csv_checked = True

    def _normalize_row(self, row):
        """Normalizes a row from the CSV file by removing extra keys."""
        return {
            key: value for key, value in row.items()
            if key in self._column_names
        }
