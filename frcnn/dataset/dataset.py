import sonnet as snt
import tensorflow as tf
import numpy as np

class Dataset(snt.AbstractModule):
    def __init__(self, config, **kwargs):
        self._cfg = config
        self._num_classes = kwargs.pop('num_classes', self._cfg.NUM_CLASSES)
        self._dataset_dir = kwargs.pop('dataset_dir', self._cfg.DATASET_DIR)
        self._num_epochs = kwargs.pop('num_epochs', self._cfg.NUM_EPOCHS)
        self._batch_size = kwargs.pop('batch_size', self._cfg.BATCH_SIZE)
        self._subset = kwargs.pop('subset', self._cfg.TRAIN_SUBSET)

        super(Dataset, self).__init__(**kwargs)
