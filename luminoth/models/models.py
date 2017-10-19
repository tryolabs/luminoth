from luminoth.models.fasterrcnn import FasterRCNN


# TODO: More models :)
MODELS = {
    'fasterrcnn': FasterRCNN,
}


def get_model(model_type):
    model_type = model_type.lower()
    if model_type not in MODELS:
        raise ValueError('"{}" is not a valid model_type'.format(model_type))

    return MODELS[model_type]
