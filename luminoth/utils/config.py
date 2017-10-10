import os.path
import tensorflow as tf
import yaml

from easydict import EasyDict


def load_config(filenames, overwrite=False, warn_overwrite=False):
    """Load config from a list of files.
    """
    if len(filenames) <= 0:
        tf.logging.error("Tried to load 0 config files.")
    config = EasyDict({})
    for filename in filenames:
        new_config = EasyDict(yaml.load(tf.gfile.GFile(filename)))
        config = merge_into(
            new_config,
            config, overwrite=overwrite, warn_overwrite=warn_overwrite
        )
    return config


def get_base_config(path, base_config_filename='base_config.yml'):
    config_path = os.path.join(os.path.dirname(path), base_config_filename)
    return load_config([config_path])


def is_basestring(value):
    """
    Checks if value is string in both Python2.7 and Python3+
    """
    return isinstance(value, (type(u''), str))


def types_compatible(new_config_value, base_config_value):
    """
    Checks that config value types are compatible.
    """
    # Allow to overwrite None values (explicit or just missing)
    if base_config_value is None:
        return True
    # Allow overwrite all None and False values.
    # TODO: reconsider this.
    if new_config_value is None or new_config_value is False:
        return True

    # Checking strings is different because in Python2 we could get different
    # types str vs unicode.
    if is_basestring(new_config_value) and is_basestring(base_config_value):
        return True

    return isinstance(new_config_value, type(base_config_value))


def merge_into(new_config, base_config, overwrite=False, warn_overwrite=False):
    """Merge one easy dict into another.

    If `overwrite` is set to true, conflicting keys will get their values from
    new_config. Else, the value will be taken from base_config.
    """
    if type(new_config) is not EasyDict:
        return

    for key, value in new_config.items():
        if value is None and base_config.get(key) is not None:
            continue

        # Since we already have the values of base_config we check against them
        if not types_compatible(value, base_config.get(key)):
            raise ValueError(
                'Incorrect type "{}" for key "{}". Must be "{}"'.format(
                    type(value), key, type(base_config.get(key))))

        # Recursively merge dicts
        if isinstance(value, dict):
            base_config[key] = merge_into(
                new_config[key], base_config.get(key, EasyDict({})),
                overwrite=overwrite, warn_overwrite=warn_overwrite
            )
        else:
            if base_config.get(key) is None:
                base_config[key] = value
            elif overwrite:
                base_config[key] = value
                if warn_overwrite:
                    tf.logging.warn('Overwrote key "{}"'.format(key))

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

    return EasyDict(override_dict)


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
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def get_model_config(base_config, custom_config, override_params):
    config = EasyDict(base_config.copy())

    if custom_config:
        # If we have a custom config file overwriting default settings
        # then we merge those values to the base_config.
        config = merge_into(custom_config, config, overwrite=True)
    if override_params:
        override_config = parse_override(override_params)
        config = merge_into(override_config, config, overwrite=True)

    return config
