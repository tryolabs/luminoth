import csv
import os
import signal
import six
import sys
import tensorflow as tf
import threading

from six.moves import queue
from PIL import Image

from luminoth.tools.dataset.readers import InvalidDataDirectory
from luminoth.tools.dataset.readers.object_detection import (
    ObjectDetectionReader
)
from luminoth.utils.dataset import read_image

CLASSES_TRAINABLE = 'classes-bbox-trainable.txt'
CLASSES_DESC = 'class-descriptions.csv'
ANNOTATIONS_FILENAME = 'annotations-human-bbox.csv'
IMAGES_LOCATION = 'gs://open-images-dataset'


class OpenImagesReader(ObjectDetectionReader):
    """OpenImagesReader supports reading the images directly from the original
    data source which is hosted in Google Cloud Storage.

    Before using it you have to request and configure access following the
    instructions here: https://github.com/cvdfoundation/open-images-dataset
    """
    def __init__(self, data_dir, split, download_threads=25, **kwargs):
        """
        Args:
            - data_dir: Path to base directory where to find all the necessary
                files and folders.
            - split: Split to use, it is used for reading the appropiate
                annotations.
            - download_threads: Number of threads to use for downloading
                images.
            - only_classes: String with classes ids to be used as filter for
                all the available classes. If the string contains ',' it will
                split the string using them.
        """
        super(OpenImagesReader, self).__init__(**kwargs)
        self._data_dir = data_dir
        self._split = split
        self._download_threads = download_threads

        self._image_ids = None

        self.yielded_records = 0
        self.errors = 0

        # Flag to notify threads if the execution is halted.
        self._alive = True

    def get_classes(self):
        trainable_labels_file = os.path.join(
            self._data_dir, CLASSES_TRAINABLE)
        trainable_labels = set()
        try:
            with tf.gfile.Open(trainable_labels_file) as tl:
                for label in tl:
                    trainable_labels.add(label.strip())
        except tf.errors.NotFoundError:
            raise InvalidDataDirectory(
                'Missing label file "{}" from data_dir'.format(
                    CLASSES_TRAINABLE))

        self.trainable_labels = self._filter_classes(trainable_labels)

        labels_descriptions_file = os.path.join(
            self._data_dir, CLASSES_DESC)
        desc_by_label = {}
        try:
            with tf.gfile.Open(labels_descriptions_file) as ld:
                reader = csv.reader(ld)
                for line in reader:
                    if line[0] in trainable_labels:
                        desc_by_label[line[0]] = line[1]
        except tf.errors.NotFoundError:
            raise InvalidDataDirectory(
                'Missing label description file "{}" from data_dir'.format(
                    CLASSES_DESC))

        self._classes = [
            desc for _, desc in
            sorted(desc_by_label.items(), key=lambda x: x[0])
        ]

    def get_total(self):
        return len(self.image_ids)

    @property
    def image_ids(self):
        if self._image_ids is None:
            annotations_file = self._get_annotations_path()
            with tf.gfile.Open(annotations_file) as af:
                reader = csv.DictReader(af)
                image_ids = set()
                for l in reader:
                    image_ids.add(l['ImageID'])
            self._image_ids = image_ids
        return self._image_ids

    def _queue_records(self, records_queue):
        """
        Read annotations from file and queue them.

        Annotations are stored in a CSV file where each line has one
        annotation. Since images can have multiple annotations (boxes), we read
        lines and merge all the annotations for one image into a single record.
        We do it this way to avoid loading the complete file in memory.

        It is VERY important that the annotation files is sorted by image_id,
        otherwise this way of reading them will not work.
        """
        annotations_file = self._get_annotations_path()
        with tf.gfile.Open(annotations_file) as af:
            reader = csv.DictReader(af)

            current_image_id = None
            partial_record = {}

            for line in reader:
                if self._stop_iteration():
                    break

                if not self._is_valid(line['ImageID']):
                    continue

                if line['ImageID'] != current_image_id:
                    # Yield if image changes and we have current image.
                    if current_image_id is not None:
                        if len(partial_record['gt_boxes']) > 0:
                            records_queue.put(partial_record)
                        else:
                            tf.logging.debug(
                                'Dropping record {} without gt_boxes.'.format(
                                    partial_record))
                            pass

                    # Start new record.
                    current_image_id = line['ImageID']

                    partial_record = {
                        'filename': current_image_id,
                        'gt_boxes': []
                    }

                # Append annotation to current record.
                try:
                    # LabelName may not exist because not all labels are
                    # trainable
                    label = self.trainable_labels.index(line['LabelName'])
                except ValueError:
                    continue

                partial_record['gt_boxes'].append({
                    'xmin': float(line['XMin']),
                    'ymin': float(line['YMin']),
                    'xmax': float(line['XMax']),
                    'ymax': float(line['YMax']),
                    'label': label,
                })

            else:
                if len(partial_record['gt_boxes']) > 0:
                    # One last yield for the last record.
                    records_queue.put(partial_record)
                else:
                    tf.logging.debug(
                        'Dropping record {} without gt_boxes.'.format(
                            partial_record))

        # Wait for all task to be consumed.
        records_queue.join()

        for _ in range(self._download_threads):
            records_queue.put(None)

    def _complete_records(self, input_queue, output_queue):
        while not self._stop_iteration():
            partial_record = input_queue.get()

            if partial_record is None:
                input_queue.task_done()
                break

            try:
                image_id = partial_record['filename']
                image_raw = read_image(
                    self._get_image_path(image_id)
                )
                image = Image.open(six.BytesIO(image_raw))

                for gt_box in partial_record['gt_boxes']:
                    gt_box['xmin'] *= image.width
                    gt_box['ymin'] *= image.height
                    gt_box['xmax'] *= image.width
                    gt_box['ymax'] *= image.height

                partial_record['width'] = image.width
                partial_record['height'] = image.height
                partial_record['depth'] = 3 if image.mode == 'RGB' else 1
                partial_record['image_raw'] = image_raw

                output_queue.put(partial_record)
            except Exception as e:
                tf.logging.error(
                    'Error processing record: {}'.format(partial_record))
                tf.logging.error(e)
                self.errors += 1
            finally:
                input_queue.task_done()

        # Notify it finished
        output_queue.put(None)

    def iterate(self):
        """
        We have a generator/consumer-generator/consumer setup where we have:
        - one thread to read the file without the images
        - multiple threads to download images and complete the records
        - the main thread to yield the completed records
        """

        signal.signal(signal.SIGINT, self._stop_reading)

        self._partial_records_queue = queue.Queue()
        generator = threading.Thread(
            target=self._queue_records,
            args=(self._partial_records_queue, )
        )
        generator.start()

        # Limit records queue to 250 because we don't want to end up with all
        # the images in memory.
        self._records_queue = queue.Queue(maxsize=250)
        consumer_threads = []
        for _ in range(self._download_threads):
            t = threading.Thread(
                target=self._complete_records,
                args=(self._partial_records_queue, self._records_queue)
            )
            t.start()
            consumer_threads.append(t)

        while not self._stop_iteration():
            record = self._records_queue.get()

            self._records_queue.task_done()
            if record is None:
                break

            self.yielded_records += 1
            yield record

        self._empty_queue(self._partial_records_queue)
        self._empty_queue(self._records_queue)

        generator.join()
        for t in consumer_threads:
            t.join()

    def _empty_queue(self, queue_to_empty):
        while not queue_to_empty.empty():
            try:
                queue_to_empty.get(False)
            except queue.Empty:
                continue
            queue_to_empty.task_done()

    def _stop_iteration(self):
        return (
            not self._alive or
            super(OpenImagesReader, self)._stop_iteration()
        )

    def _stop_reading(self, signal, frame):
        self._alive = False
        sys.exit(1)

    def _get_annotations_path(self):
        return os.path.join(
            self._data_dir, self._split, ANNOTATIONS_FILENAME
        )

    def _get_image_path(self, image_id):
        return os.path.join(
            IMAGES_LOCATION, self._split, '{}.jpg'.format(image_id)
        )
