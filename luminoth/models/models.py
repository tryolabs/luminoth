from .fasterrcnn import FasterRCNN
from .pretrained import VGG, ResNetV2


MODELS = {
    'fasterrcnn': FasterRCNN,
}

PRETRAINED_MODELS = {
    'vgg': VGG,
    'vgg_16': VGG,
    'resnet': ResNetV2,
    'resnetv2': ResNetV2,
}
