import datetime
import json
import os.path
import subprocess
import tensorflow as tf


DEFAULT_BASE_PATH = os.path.expanduser('~/.luminoth')
DEFAULT_FILENAME = 'runs.json'
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_diff():
    try:
        return subprocess.check_output(
            ['git', 'diff'], cwd=CURRENT_DIR
        ).strip().decode('utf-8')
    except:  # noqa
        # Never fail, we don't care about the error.
        return None


def get_luminoth_version():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=CURRENT_DIR
        ).strip().decode('utf-8')
    except:  # noqa
        # Never fail, we don't care about the error.
        pass

    try:
        from luminoth import __version__ as lumi_version
        return lumi_version
    except ImportError:
        pass


def get_tensorflow_version():
    try:
        from tensorflow import __version__ as tf_version
        return tf_version
    except ImportError:
        pass


def save_run(config, environment=None, comment=None, extra_config=None,
             base_path=DEFAULT_BASE_PATH, filename=DEFAULT_FILENAME):
    if environment == 'cloud':
        # We don't write runs inside Google Cloud, we run it before.
        return

    diff = get_diff()
    lumi_version = get_luminoth_version()
    tf_version = get_tensorflow_version()

    experiment = {
        'environment': environment,
        'datetime': str(datetime.datetime.utcnow()) + 'Z',
        'diff': diff,
        'luminoth_version': lumi_version,
        'tensorflow_version': tf_version,
        'config': config,
        'extra_config': extra_config,
    }

    file_path = os.path.join(base_path, filename)
    tf.gfile.MakeDirs(base_path)

    with tf.gfile.Open(file_path, 'a') as log:
        log.write(json.dumps(experiment) + '\n')
