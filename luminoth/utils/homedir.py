"""Luminoth home (~/.luminoth) management utilities."""
import os
import tensorflow as tf


DEFAULT_LUMINOTH_HOME = os.path.expanduser('~/.luminoth')


def get_luminoth_home(create_if_missing=True):
    """Returns Luminoth's homedir."""
    # Get Luminoth's home directory (the default one or the overridden).
    path = DEFAULT_LUMINOTH_HOME
    if 'LUMI_HOME' in os.environ:
        path = os.environ['LUMI_HOME']
    path = os.path.abspath(path)

    # Create the directory if it doesn't exist.
    if create_if_missing and not os.path.exists(path):
        tf.gfile.MakeDirs(path)

    return path
