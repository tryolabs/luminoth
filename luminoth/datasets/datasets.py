from luminoth.datasets.tfrecord import TFRecordDataset

DATASETS = {
    'tfrecord': TFRecordDataset,
}


def get_dataset(dataset_type):
    dataset_type = dataset_type.lower()
    if dataset_type not in DATASETS:
        raise ValueError('"{}" is not a valid dataset_type'
                         .format(dataset_type))

    return DATASETS[dataset_type]
