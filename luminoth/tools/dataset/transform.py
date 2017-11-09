import os
import click
import tensorflow as tf

from luminoth.datasets.exceptions import InvalidDataDirectory

from .readers import get_reader, READERS
from .writers import ObjectDetectionWriter


def get_output_subfolder(only_filename, limit_examples, limit_classes):
    """
    Returns: subfolder name for records
    """
    if only_filename is not None:
        return 'only-{}'.format(only_filename)
    elif limit_examples is not None and limit_classes is not None:
        return 'limit-{}-classes-{}'.format(limit_examples, limit_classes)
    elif limit_examples is not None:
        return 'limit-{}'.format(limit_examples)
    elif limit_classes is not None:
        return 'classes-{}'.format(limit_classes)


@click.command()
@click.option('dataset_reader', '--type', type=click.Choice(READERS.keys()))  # noqa
@click.option('--data-dir', help='Where to locate the original data.')
@click.option('--output-dir', help='Where to save the transformed data.')
@click.option('splits', '--split', multiple=True, help='Which splits to transform.')  # noqa
@click.option('--only-filename', help='Create dataset with a single example.')
@click.option('--limit-examples', type=int, help='Limit dataset with to the first `N` examples.')  # noqa
@click.option('--limit-classes', type=int, help='Limit dataset with `N` random classes.')  # noqa
@click.option('--seed', type=int, help='Seed used for picking random classes.')
@click.option('--debug', is_flag=True, help='Set level logging to DEBUG.')
def transform(dataset_reader, data_dir, output_dir, splits, only_filename,
              limit_examples, limit_classes, seed, debug):
    """
    Prepares dataset for ingestion.

    Converts the dataset into different (one per split) TFRecords files.
    """
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    # We forcefully save modified datasets into subfolders to avoid
    # overwriting and/or unnecessary clutter.
    output_subfolder = get_output_subfolder(
        only_filename, limit_examples, limit_classes
    )
    if output_subfolder:
        output_dir = os.path.join(output_dir, output_subfolder)

    try:
        reader = get_reader(dataset_reader)
    except ValueError as e:
        tf.logging.error('Error getting reader: {}'.format(e))
        return

    # All splits must have a consistent set of classes.
    classes = None

    try:
        for split in splits:
            # Create instance of reader.
            split_reader = reader(
                data_dir, split,
                only_filename=only_filename, limit_examples=limit_examples,
                limit_classes=limit_classes, seed=seed
            )

            if classes is None:
                # "Save" classes from the first split reader
                classes = split_reader.classes
            else:
                # Overwrite classes after first split for consistency.
                split_reader.set_classes(classes)

            # We assume we are saving object detection objects, but it should
            # be easy to modify once we have different types of objects.
            writer = ObjectDetectionWriter(split_reader, output_dir, split)
            writer.save()
    except InvalidDataDirectory as e:
        tf.logging.error('Error reading dataset: {}'.format(e))
