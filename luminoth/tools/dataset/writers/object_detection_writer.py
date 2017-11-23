import click
import json
import os
import tensorflow as tf

from .base_writer import BaseWriter

from luminoth.tools.dataset.readers import ObjectDetectionReader
from luminoth.utils.dataset import to_int64, to_string, to_bytes

REQUIRED_KEYS = set(
    ['width', 'height', 'depth', 'filename', 'image_raw', 'gt_boxes']
)
REQUIRED_GT_KEYS = set(['label', 'xmin', 'ymin', 'xmax', 'ymax'])

CLASSES_FILENAME = 'classes.json'


class InvalidRecord(Exception):
    pass


class ObjectDetectionWriter(BaseWriter):
    """Writes object detection dataset into tfrecords.

    Reads dataset from a subclass of ObjectDetectionReader and saves it using
    the default format for tfrecords.
    """
    def __init__(self, reader, output_dir, split='data'):
        """
        Args:
            reader:
            output_dir: Directory to save the resulting tfrecords.
            split: Split being save, which is used as a filename for the
                resulting file.
        """
        super(ObjectDetectionWriter, self).__init__()
        if not isinstance(reader, ObjectDetectionReader):
            raise ValueError(
                'Saver needs a valid ObjectDetectionReader subclass'
            )

        self._reader = reader
        self._output_dir = output_dir
        self._split = split

    def save(self):
        """
        """
        tf.logging.info('Saving split "{}" in output_dir = {}'.format(
            self._split, self._output_dir))
        if not tf.gfile.Exists(self._output_dir):
            tf.gfile.MakeDirs(self._output_dir)

        # Save classes in simple json format for later use.
        classes_file = os.path.join(self._output_dir, CLASSES_FILENAME)
        json.dump(self._reader.classes, tf.gfile.GFile(classes_file, 'w'))
        record_file = os.path.join(
            self._output_dir, '{}.tfrecords'.format(self._split))
        writer = tf.python_io.TFRecordWriter(record_file)

        tf.logging.debug('Found {} images.'.format(self._reader.total))

        with click.progressbar(self._reader.iterate(),
                               length=self._reader.total) as record_list:
            for record_idx, record in enumerate(record_list):
                tf_record = self._record_to_tf(record)
                if tf_record is not None:
                    writer.write(tf_record.SerializeToString())

            if self._output_dir.startswith('gs://'):
                tf.logging.info('Saving tfrecord to Google Cloud Storage. '
                                'It may take a while.')
            writer.close()

        if self._reader.yielded_records == 0:
            tf.logging.error(
                'Data is missing. Removing record file. '
                '(Use "--debug" flag to display all logs)')
            tf.gfile.Remove(record_file)
            return
        elif self._reader.errors > 0:
            tf.logging.warning(
                'Failed on {} records. '
                '(Use "--debug" flag to display all logs)'.format(
                    self._reader.errors, self._reader.yielded_records
                )
            )

        tf.logging.info('Saved {} records to "{}"'.format(
            self._reader.yielded_records, record_file))

    def _validate_record(self, record):
        """
        Checks that the record is valid before saving it.

        Args:
            record: `dict`

        Raises:
            InvalidRecord when required keys are missing from the record.
        """
        record_keys = set(record.keys())
        if record_keys != REQUIRED_KEYS:
            raise InvalidRecord('Missing keys: {}'.format(
                REQUIRED_KEYS - record_keys
            ))

        if len(record['gt_boxes']) == 0:
            raise InvalidRecord('Record should have at least one `gt_boxes`')

        for gt_box in record['gt_boxes']:
            gt_keys = set(gt_box.keys())
            if gt_keys != REQUIRED_GT_KEYS:
                raise InvalidRecord('Missing gt boxes keys {}'.format(
                    REQUIRED_GT_KEYS - gt_keys
                ))

    def _record_to_tf(self, record):
        """Creates tf.train.SequenceExample object from records.
        """
        try:
            self._validate_record(record)
        except InvalidRecord as e:
            # Pop image before displaying record.
            record.pop('image_raw')
            tf.logging.warning('Invalid record: {} - {}'.format(
                e, record
            ))
            return

        sequence_vals = {
            'label': [],
            'xmin': [],
            'ymin': [],
            'xmax': [],
            'ymax': [],
        }

        for b in record['gt_boxes']:
            sequence_vals['label'].append(to_int64(b['label']))
            sequence_vals['xmin'].append(to_int64(b['xmin']))
            sequence_vals['ymin'].append(to_int64(b['ymin']))
            sequence_vals['xmax'].append(to_int64(b['xmax']))
            sequence_vals['ymax'].append(to_int64(b['ymax']))

        object_feature_lists = {
            'label': tf.train.FeatureList(feature=sequence_vals['label']),
            'xmin': tf.train.FeatureList(feature=sequence_vals['xmin']),
            'ymin': tf.train.FeatureList(feature=sequence_vals['ymin']),
            'xmax': tf.train.FeatureList(feature=sequence_vals['xmax']),
            'ymax': tf.train.FeatureList(feature=sequence_vals['ymax']),
        }

        object_features = tf.train.FeatureLists(
            feature_list=object_feature_lists
        )

        feature = {
            'width': to_int64(record['width']),
            'height': to_int64(record['height']),
            'depth': to_int64(record['depth']),
            'filename': to_string(record['filename']),
            'image_raw': to_bytes(record['image_raw']),
        }

        # Now build an `Example` protobuf object and save with the writer.
        context = tf.train.Features(feature=feature)
        example = tf.train.SequenceExample(
            feature_lists=object_features, context=context
        )

        return example
