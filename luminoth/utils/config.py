import os.path
import yaml
import easydict


def load_config(file):
    return easydict.EasyDict(yaml.load(file))


def get_base_config(path, base_config_filename='base_config.yml'):
    config_path = os.path.join(os.path.dirname(path), base_config_filename)
    return easydict.EasyDict(yaml.load(open(config_path)))


def kwargs_to_config(kwargs):
    return easydict.EasyDict(dict(
        (key, val)
        for key, val in kwargs.items()
        if val is not None
    ))


def merge_into(new_config, base_config):
    if type(new_config) is not easydict.EasyDict:
        return

    for key, value in new_config.items():
        # All keys in new_config must be overwriting values in base_config
        if key not in base_config:
            raise KeyError('Key "{}" is not valid.'.format(key))

        # Since we already have the values of base_config we check against them
        if (base_config[key] is not None and
           type(base_config[key]) is not type(value)):
            raise ValueError(
                'Incorrect type "{}" for key "{}". Must be "{}"'.format(
                    type(value), key, type(base_config[key])))

        # Recursively merge dicts
        if type(value) is easydict.EasyDict:
            try:
                merge_into(new_config[key], base_config[key])
            except (KeyError, ValueError) as e:
                raise e
        else:
            base_config[key] = value

    return base_config
