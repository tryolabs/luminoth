import sys
import click
import tensorflow as tf

from .dataset import RecordSaver, InvalidDataDirectory
from .imagenet import ImageNet
from .pascalvoc import PascalVOC

VALID_DATASETS = {
    'voc': PascalVOC,
    'pascalvoc': PascalVOC,
    'imagenet': ImageNet,
}


@click.command()
@click.option('dataset_type', '--type', type=click.Choice(VALID_DATASETS.keys()))  # noqa
@click.option('--data-dir', default='datasets/')
@click.option('--output-dir', default='datasets/tf')
@click.option('ignore_splits', '--ignore-split', multiple=True)
@click.option('--only-filename', help='Create dataset with a single example.')
@click.option('--limit-examples', type=int, help='Limit dataset with to the first `N` examples.')  # noqa
@click.option('--limit-classes', type=int, help='Limit dataset with `N` random classes.')  # noqa
@click.option('--seed', type=int, help='Seed used for picking random classes.')
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def transform(dataset_type, data_dir, output_dir, ignore_splits, only_filename,
              limit_examples, limit_classes, seed, debug):
    """
    Prepares dataset for ingestion.

    Converts the dataset into different (one per split) TFRecords files.
    """
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    try:
        ds = VALID_DATASETS[dataset_type](data_dir=data_dir)
    except InvalidDataDirectory as e:
        tf.logging.error('Invalid data directory: {}'.format(e))
        sys.exit(1)

    saver = RecordSaver(
        ds, output_dir,
        ignore_splits=ignore_splits,
        only_filename=only_filename,
        limit_examples=limit_examples,
        limit_classes=limit_classes,
        seed=seed
    )

    saver.save()
