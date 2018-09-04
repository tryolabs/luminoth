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

# Compatible with OpenImages V4
# Files available at: https://storage.googleapis.com/openimages/web/index.html
CLASSES_TRAINABLE = '{split}-annotations-human-imagelabels-boxable.csv'
ANNOTATIONS_FILENAME = '{split}-annotations-bbox.csv'
CLASSES_DESC = 'class-descriptions-boxable.csv'
IMAGES_LOCATION = 's3://open-images-dataset'


class OpenImagesReader(ObjectDetectionReader):
    """OpenImagesReader supports reading the images directly from the original
    data source which is hosted in Google Cloud Storage.

    Before using it you have to request and configure access following the
    instructions here: https://github.com/cvdfoundation/open-images-dataset
    """
    def __init__(self, data_dir, split, download_threads=25, **kwargs):
        """
        Args:
            data_dir: Path to base directory where to find all the necessary
                files and folders.
            split: Split to use, it is used for reading the appropiate
                annotations.
            download_threads: Number of threads to use for downloading
                images.
            only_classes: String with classes ids to be used as filter for
                all the available classes. If the string contains ',' it will
                split the string using them.
        """
        super(OpenImagesReader, self).__init__(**kwargs)
        self._data_dir = data_dir
        self._split = split
        self._download_threads = download_threads

        self._image_ids = None
        self.desc_by_label = {}

        self.yielded_records = 0
        self.errors = 0
        self._total_queued = 0
        # Flag to notify threads if the execution is halted.
        self._alive = True

    def _get_classes_path(self):
        """
        Return the full path to the CLASSES_TRAINABLE for the current split
        in the data directory.

        We expect this file to be located in a directory corresponding to the
        split, ie. "train", "validation", "test".
        """
        return os.path.join(
            self._data_dir, self._split, CLASSES_TRAINABLE
        ).format(split=self._split)

    def _get_annotations_path(self):
        """
        Return the full path to the ANNOTATIONS_FILENAME for the current split
        in the data directory.

        We expect this file to be located in a directory corresponding to the
        split, ie. "train", "validation", "test".
        """
        return os.path.join(
            self._data_dir, self._split, ANNOTATIONS_FILENAME
        ).format(split=self._split)

    def _get_image_path(self, image_id):
        return os.path.join(
            IMAGES_LOCATION, self._split, '{}.jpg'.format(image_id)
        ).format(split=self._split)

    def get_classes(self):
        trainable_labels_file = self._get_classes_path()
        trainable_labels = set()
        try:
            with tf.gfile.Open(trainable_labels_file) as tl:
                reader = csv.reader(tl)
                # Skip header
                next(reader, None)
                for line in reader:
                    trainable_labels.add(line[2])
        except tf.errors.NotFoundError:
            raise InvalidDataDirectory(
                'The label file "{}" must be in the root data '
                'directory: {}'.format(
                    os.path.split(trainable_labels_file)[1], self._data_dir
                )
            )

        self.trainable_labels = self._filter_classes(trainable_labels)

        # Build the map from classes to description for pretty printing their
        # names.
        labels_descriptions_file = os.path.join(self._data_dir, CLASSES_DESC)
        try:
            with tf.gfile.Open(labels_descriptions_file) as ld:
                reader = csv.reader(ld)
                for line in reader:
                    if line[0] in self.trainable_labels:
                        self.desc_by_label[line[0]] = line[1]
        except tf.errors.NotFoundError:
            raise InvalidDataDirectory(
                'Missing label description file "{}" from root data '
                'directory: {}'.format(CLASSES_DESC, self._data_dir)
            )

        return self.trainable_labels

    def pretty_name(self, label):
        return '{} ({})'.format(self.desc_by_label[label], label)

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

    def _queue_record(self, queue, record):
        if not record['gt_boxes']:
            tf.logging.debug(
                'Dropping record {} without gt_boxes.'.format(record))
            return

        # If asking for a limited number per class, only yield if the current
        # example adds at least 1 new class that hasn't been maxed out. For
        # example, if "Persons" has been maxed out but "Bus" has not, a new
        # image containing only instances of "Person" will not be yielded,
        # while an image containing both "Person" and "Bus" instances will.
        if self._class_examples:
            labels_in_image = set([
                self.classes[bbox['label']] for bbox in record['gt_boxes']
            ])
            not_maxed_out = labels_in_image - self._maxed_out_classes

            if not not_maxed_out:
                tf.logging.debug(
                    'Dropping record {} with maxed-out labels: {}'.format(
                        record['filename'], labels_in_image))
                return

            tf.logging.debug(
                'Queuing record {} with labels: {}'.format(
                    record['filename'], labels_in_image))

        self._will_add_record(record)
        queue.put(record)

    def _queue_partial_records(self, partial_records_queue, records_queue):
        """
        Read annotations from file and queue them.

        Annotations are stored in a CSV file where each line has one
        annotation. Since images can have multiple annotations (boxes), we read
        lines and merge all the annotations for one image into a single record.
        We do it this way to avoid loading the complete file in memory.

        It is VERY important that the annotation file is sorted by image_id,
        otherwise this way of reading them will not work.
        """
        annotations_file = self._get_annotations_path()

        with tf.gfile.Open(annotations_file) as af:
            reader = csv.DictReader(af)

            current_image_id = None
            partial_record = {}

            # Number of records we have queued so far, which should be
            # completed by another thread.
            num_queued_records = 0

            for line in reader:
                if num_queued_records == self.total:
                    # Reached the max number of records we can or want to
                    # process.
                    break

                if self._all_maxed_out():
                    break

                if self._should_skip(line['ImageID']):
                    continue

                # Filter group annotations (we only want single instances)
                if line['IsGroupOf'] == '1':
                    continue

                # Append annotation to current record.
                try:
                    # LabelName may not exist because not all labels are
                    # trainable
                    label = self.trainable_labels.index(line['LabelName'])
                except ValueError:
                    continue

                if line['ImageID'] != current_image_id:
                    # Yield if image changes and we have current image.
                    if current_image_id is not None:
                        num_queued_records += 1
                        self._queue_record(
                            partial_records_queue,
                            partial_record
                        )

                    # Start new record.
                    current_image_id = line['ImageID']
                    partial_record = {
                        'filename': current_image_id,
                        'gt_boxes': []
                    }

                partial_record['gt_boxes'].append({
                    'xmin': float(line['XMin']),
                    'ymin': float(line['YMin']),
                    'xmax': float(line['XMax']),
                    'ymax': float(line['YMax']),
                    'label': label,
                })

            else:
                # No data we care about in dataset -- nothing to queue
                if partial_record:
                    num_queued_records += 1
                    self._queue_record(
                        partial_records_queue,
                        partial_record
                    )

        tf.logging.debug('Stopped queuing records.')

        # Wait for all records to be consumed by the threads that complete them
        partial_records_queue.join()

        tf.logging.debug('All records consumed!')

        # Signal the main thread that we have finished producing and every
        # record in the the queues has been consumed.
        records_queue.put(None)

    def _complete_records(self, input_queue, output_queue):
        """
        Daemon thread that will complete queued records from `input_queue` and
        put them in `output_queue`, where they will be read and yielded by the
        main thread.

        This is the thread that will actually download the images of the
        dataset.
        """
        while True:
            try:
                partial_record = input_queue.get()

                image_id = partial_record['filename']
                image_raw = read_image(self._get_image_path(image_id))
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

    def iterate(self):
        """
        We have a generator/consumer-generator/consumer setup where we have:
        - one thread to read the file without the images
        - multiple threads to download images and complete the records
        - the main thread to yield the completed records
        """
        signal.signal(signal.SIGINT, self._stop_reading)

        # Which records to complete (missing image)
        partial_records_queue = queue.Queue()

        # Limit records queue to 250 because we don't want to end up with all
        # the images in memory.
        records_queue = queue.Queue(maxsize=250)

        generator = threading.Thread(
            target=self._queue_partial_records,
            args=(partial_records_queue, records_queue)
        )
        generator.start()

        for _ in range(self._download_threads):
            t = threading.Thread(
                target=self._complete_records,
                args=(partial_records_queue, records_queue)
            )
            t.daemon = True
            t.start()

        while not self._stop_iteration():
            record = records_queue.get()

            if record is None:
                break

            self.yielded_records += 1
            yield record

        # In case we were killed by signal
        self._empty_queue(partial_records_queue)
        self._empty_queue(records_queue)

        # Wait for generator to finish peacefuly...
        generator.join()

    def _empty_queue(self, queue_to_empty):
        while not queue_to_empty.empty():
            try:
                queue_to_empty.get(False)
            except queue.Empty:
                continue
            queue_to_empty.task_done()

    def _stop_iteration(self):
        """
        Override the parent implementation, because we deal with this in the
        producer thread.
        """
        if not self._alive:
            return True

    def _stop_reading(self, signal, frame):
        self._alive = False
        sys.exit(1)
