import sonnet as snt
import tensorflow as tf
import numpy as np

class Dataset(snt.AbstractModule):

    NUM_CLASSES = 20  # default to NUM_CLASSES of PASCAL VOC
    DATASET_DIR = 'datasets/voc/tf'

    def __init__(self, *args, **kwargs):
        self._num_classes = kwargs.pop('num_classes', self.NUM_CLASSES)
        self._dataset_dir = kwargs.pop('dataset_dir', self.DATASET_DIR)
        self._num_epochs = kwargs.pop('num_epochs', 10)
        self._batch_size = kwargs.pop('batch_size', 1)
        self._subset = kwargs.pop('subset', 'train')
        self._losses = {}

        super(Dataset, self).__init__(*args, **kwargs)


    def cost(self, rpn_bbox_prediction, rpn_bbox_target):
        """
        Returns cost for general object detection dataset.

        TODO: Maybe we need many methods, one for each

        Args:
            TODO:

        Returns:
            Multi-loss?
        """
        pass