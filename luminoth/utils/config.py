import easydict
import os.path
import tensorflow as tf
import yaml


def load_config(filename):
    return easydict.EasyDict(yaml.load(tf.gfile.GFile(filename)))


def get_base_config(path, base_config_filename='base_config.yml'):
    config_path = os.path.join(os.path.dirname(path), base_config_filename)
    return load_config(config_path)


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
            raise KeyError('Key "{}" is not a valid config key.'.format(key))

        # Since we already have the values of base_config we check against them
        # if (base_config[key] is not None and
        #    type(base_config[key]) is not type(value)):
        #     raise ValueError(
        #         'Incorrect type "{}" for key "{}". Must be "{}"'.format(
        #             type(value), key, type(base_config[key])))

        # Recursively merge dicts
        if type(value) is easydict.EasyDict:
            try:
                merge_into(new_config[key], base_config[key])
            except (KeyError, ValueError) as e:
                raise e
        else:
            base_config[key] = value

    return base_config


def parse_override(override_options):
    if not override_options:
        return {}

    override_dict = {}
    for option in override_options:
        key_value = option.split('=')
        if len(key_value) != 2:
            raise ValueError('Invalid override option "{}"'.format(option))
        key, value = key_value
        nested_keys = key.split('.')

        local_override_dict = override_dict
        for nested_key in nested_keys[:-1]:
            if nested_key not in local_override_dict:
                local_override_dict[nested_key] = {}
            local_override_dict = local_override_dict[nested_key]

        local_override_dict[nested_keys[-1]] = parse_config_value(value)

    return easydict.EasyDict(override_dict)


def parse_config_value(value):
    """
    Try to parse the config value to boolean, integer, float or string.
    We assume all values are strings.
    """
    if value == '':
        return None
    elif value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False

    try:
        return int(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    return value
