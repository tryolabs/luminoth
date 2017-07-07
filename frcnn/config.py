
class Config:
    ANCHOR_SCALES = [0.5, 1, 2]
    ANCHOR_RATIOS = [0.5, 1, 2]
    ANCHOR_BASE_SIZE = 256
    ANCHOR_STRIDE = 16

    # Dataset
    NUM_CLASSES = 20
    DATASET_DIR = 'datasets/voc/tf'
    NUM_EPOCHS = 10
    BATCH_SIZE = 1
    TRAIN_SUBSET = 'train'

    IMAGE_MIN_SIZE = 600
    IMAGE_MAX_SIZE = 1024

    # Pretrained
    PRETRAINED_TRAINABLE = True
