from .base_reader import BaseReader, InvalidDataDirectory  # noqa
from .object_detection import ObjectDetectionReader  # noqa
from .object_detection import (
    COCOReader, CSVReader, FlatReader, ImageNetReader, OpenImagesReader,
    PascalVOCReader,
)

READERS = {
    'coco': COCOReader,
    'csv': CSVReader,
    'flat': FlatReader,
    'imagenet': ImageNetReader,
    'openimages': OpenImagesReader,
    'pascal': PascalVOCReader,
}


def get_reader(reader):
    reader = reader.lower()
    if reader not in READERS:
        raise ValueError('"{}" is not a valid reader'.format(reader))

    return READERS[reader]
