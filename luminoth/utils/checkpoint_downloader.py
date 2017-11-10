import click
import json
import os
import requests
import tarfile
import tensorflow as tf


TENSORFLOW_OFFICIAL_ENDPOINT = 'http://download.tensorflow.org/models/'

BASE_NETWORK_FILENAMES = {
    'inception_v3': 'inception_v3_2016_08_28.tar.gz',
    'resnet_v1_50': 'resnet_v1_50_2016_08_28.tar.gz',
    'resnet_v1_101': 'resnet_v1_101_2016_08_28.tar.gz',
    'resnet_v1_152': 'resnet_v1_152_2016_08_28.tar.gz',
    'resnet_v2_50': 'resnet_v2_50_2017_04_14.tar.gz',
    'resnet_v2_101': 'resnet_v2_101_2017_04_14.tar.gz',
    'resnet_v2_152': 'resnet_v2_152_2017_04_14.tar.gz',
    'vgg_16': 'vgg_16_2016_08_28.tar.gz',
    'vgg_19': 'vgg_19_2016_08_28.tar.gz',
}


def get_default_path():
    if 'TF_CONFIG' in os.environ:
        tf_config = json.loads(os.environ['TF_CONFIG'])
        job_dir = tf_config.get('job', {}).get('job_dir')
        if job_dir:
            # Instead of using the job_dir we create a folder inside.
            job_dir = os.path.join(job_dir, 'pretrained_checkpoints/')
            return job_dir

    return '~/.luminoth/'


DEFAULT_PATH = get_default_path()


def get_checkpoint_path(path=DEFAULT_PATH):
    # Expand user if path is relative to user home.
    path = os.path.expanduser(path)

    if not path.startswith('gs://'):
        # We don't need to create Google cloud storage "folders"
        path = os.path.abspath(path)

    if not tf.gfile.Exists(path):
        tf.logging.info(
            'Creating folder "{}" to save checkpoints.'.format(path))
        tf.gfile.MakeDirs(path)

    return path


def download_checkpoint(network, network_filename, checkpoint_path,
                        checkpoint_filename):
    tarball_filename = BASE_NETWORK_FILENAMES[network]
    url = TENSORFLOW_OFFICIAL_ENDPOINT + tarball_filename
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('Content-Length'))
    tarball_path = os.path.join(checkpoint_path, tarball_filename)
    tmp_tarball = tf.gfile.Open(tarball_path, 'wb')
    tf.logging.info('Downloading {} checkpoint.'.format(network_filename))
    with click.progressbar(length=total_size) as bar:
        for data in response.iter_content(chunk_size=4096):
            tmp_tarball.write(data)
            bar.update(len(data))
    tmp_tarball.flush()

    tf.logging.info('Saving checkpoint to {}'.format(checkpoint_path))
    # Open saved tarball as readable binary
    tmp_tarball = tf.gfile.Open(tarball_path, 'rb')
    # Open tarfile object
    tar_obj = tarfile.open(fileobj=tmp_tarball)
    # Create buffer with extracted network checkpoint
    checkpoint_fp = tar_obj.extractfile(network_filename)
    # Define where to save.
    checkpoint_file = tf.gfile.Open(checkpoint_filename, 'wb')
    # Write extracted checkpoint to file
    checkpoint_file.write(checkpoint_fp.read())
    checkpoint_file.flush()
    checkpoint_file.close()
    tmp_tarball.close()
    # Remove temp tarball
    tf.gfile.Remove(tarball_path)


def get_checkpoint_file(network, checkpoint_path=DEFAULT_PATH):
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_PATH
    checkpoint_path = get_checkpoint_path(path=checkpoint_path)
    files = tf.gfile.ListDirectory(checkpoint_path)
    network_filename = '{}.ckpt'.format(network)
    checkpoint_file = os.path.join(checkpoint_path, network_filename)
    if network_filename not in files:
        download_checkpoint(
            network, network_filename, checkpoint_path, checkpoint_file
        )

    return checkpoint_file
