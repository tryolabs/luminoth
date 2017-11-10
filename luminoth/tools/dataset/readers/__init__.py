from .base_reader import BaseReader, InvalidDataDirectory  # noqa
from .object_detection import ObjectDetectionReader  # noqa
from .object_detection import (
    ImageNetReader, PascalVOCReader, CSVReader, COCOReader
)

READERS = {
    'imagenet': ImageNetReader,
    'pascal': PascalVOCReader,
    'voc': PascalVOCReader,
    'csv': CSVReader,
    'coco': COCOReader,
}


def get_reader(reader):
    reader = reader.lower()
    if reader not in READERS:
        raise ValueError('"{}" is not a valid reader'.format(reader))

    return READERS[reader]
