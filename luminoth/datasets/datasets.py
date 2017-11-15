import tensorflow as tf

from luminoth.datasets.object_detection_dataset import ObjectDetectionDataset

DATASETS = {
    'tfrecord': ObjectDetectionDataset,
    'object_detection': ObjectDetectionDataset,
}


def get_dataset(dataset_type):
    dataset_type = dataset_type.lower()
    if dataset_type not in DATASETS:
        raise ValueError('"{}" is not a valid dataset_type'
                         .format(dataset_type))

    if dataset_type == 'tfrecord':
        tf.logging.warning(
            'Dataset `tfrecord` is deprecated. Use `object_detection` instead.'
        )

    return DATASETS[dataset_type]
