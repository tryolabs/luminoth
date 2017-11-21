import csv
import os
import six
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
        super(OpenImagesReader, self).__init__()
        self._data_dir = data_dir
        self._split = split
        self._download_threads = download_threads

        self._classes = None
        self._total = None
        self._image_ids = None

        self.yielded_records = 0
        self.errors = 0

    @property
    def classes(self):
        if self._classes is None:
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

            self._classes = sorted(trainable_labels)
            self._descriptions = [
                desc for _, desc in
                sorted(desc_by_label.items(), key=lambda x: x[0])
            ]

        return self._classes

    @property
    def total(self):
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
        annotations_file = self._get_annotations_path()
        with tf.gfile.Open(annotations_file) as af:
            reader = csv.DictReader(af)

            current_image_id = None
            partial_record = {}

            for line in reader:
                if line['ImageID'] != current_image_id:
                    # Yield if image changes and we have current image.
                    if current_image_id is not None:
                        records_queue.put(partial_record)

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
                    label = self.classes.index(line['LabelName'])
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
                # One last yield for the last record.
                records_queue.put(partial_record)

        # Wait for all task to be consumed.
        records_queue.join()

        for _ in range(self._download_threads):
            records_queue.put(None)

    def _complete_records(self, input_queue, output_queue):
        while True:
            partial_record = input_queue.get()

            if partial_record is None:
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

                input_queue.task_done()
                output_queue.put(partial_record)
            except Exception as e:
                tf.logging.warning(
                    'Error processing record: {}'.format(partial_record))
                tf.logging.error(e)
                self.errors += 1

        # Notify it finished
        output_queue.put(None)

    def iterate(self):
        """
        We have a generator/consumer-generator/consumer setup where we have:
        - one thread to read the file without the images
        - multiple threads to download images and complete the records
        - the main thread to yield the completed records
        """
        partial_records_queue = queue.Queue()
        generator = threading.Thread(
            target=self._queue_records,
            args=(partial_records_queue, )
        )
        generator.start()

        # Limit records queue to 250 because we don't want to end up with all
        # the images in memory.
        records_queue = queue.Queue(maxsize=250)
        consumer_threads = []
        for _ in range(self._download_threads):
            t = threading.Thread(
                target=self._complete_records,
                args=(partial_records_queue, records_queue)
            )
            t.start()
            consumer_threads.append(t)

        while True:
            record = records_queue.get()
            if record is None:
                break

            self.yielded_records += 1
            yield record
            records_queue.task_done()

        generator.join()
        for t in consumer_threads:
            t.join()

    def _get_annotations_path(self):
        return os.path.join(
            self._data_dir, self._split, ANNOTATIONS_FILENAME
        )

    def _get_image_path(self, image_id):
        return os.path.join(
            IMAGES_LOCATION, self._split, '{}.jpg'.format(image_id)
        )
