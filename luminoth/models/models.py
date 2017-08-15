from luminoth.models.fasterrcnn import FasterRCNN
from luminoth.models.pretrained import VGG, ResNetV2


MODELS = {
    'fasterrcnn': FasterRCNN,
    'vgg': VGG,
    'vgg_16': VGG,
    'resnet': ResNetV2,
    'resnetv2': ResNetV2,
}


def get_model(model_type):
    model_type = model_type.lower()
    if model_type not in MODELS:
        raise ValueError('"{}" is not a valid model_type'.format(model_type))

    return MODELS[model_type]
