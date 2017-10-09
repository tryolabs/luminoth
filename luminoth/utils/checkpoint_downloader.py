import tensorflow as tf
import click
import requests
import os
import tempfile
import tarfile

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

DEFAULT_PATH = '~/.luminoth/'


def get_checkpoint_path(path=DEFAULT_PATH):
    tf.logging.debug('Creating folder "{}" to save checkpoints.'.format(path))
    full_path = os.path.abspath(os.path.expanduser(path))
    tf.gfile.MakeDirs(full_path)
    return full_path


def download_checkpoint(network, network_filename, checkpoint_path):
    url = TENSORFLOW_OFFICIAL_ENDPOINT + BASE_NETWORK_FILENAMES[network]
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('Content-Length'))
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tf.logging.info('Downloading {} checkpoint.'.format(network_filename))
    with click.progressbar(length=total_size) as bar:
        for data in response.iter_content(chunk_size=4096):
            tmp_file.write(data)
            bar.update(len(data))
    tmp_file.flush()

    tf.logging.info('Saving checkpoint to {}'.format(checkpoint_path))
    tar_obj = tarfile.open(tmp_file.name)
    tar_obj.extract(network_filename, checkpoint_path)
    tmp_file.close()


def get_checkpoint_file(network, checkpoint_path=DEFAULT_PATH):
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_PATH
    checkpoint_path = get_checkpoint_path(path=checkpoint_path)
    files = tf.gfile.ListDirectory(checkpoint_path)
    network_filename = '{}.ckpt'.format(network)
    network_file = os.path.join(checkpoint_path, network_filename)
    if network_filename not in files:
        download_checkpoint(network, network_filename, checkpoint_path)

    return network_file
