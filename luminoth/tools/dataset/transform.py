import click
import tensorflow as tf

from luminoth.datasets.exceptions import InvalidDataDirectory
from luminoth.utils.config import parse_override
from .readers import get_reader, READERS
from .writers import ObjectDetectionWriter


@click.command()
@click.option('dataset_reader', '--type', type=click.Choice(READERS.keys()), required=True)  # noqa
@click.option('--data-dir', required=True, help='Where to locate the original data.')  # noqa
@click.option('--output-dir', required=True, help='Where to save the transformed data.')  # noqa
@click.option('splits', '--split', required=True, multiple=True, help='The splits to transform (ie. train, test, val).')  # noqa
@click.option('--only-classes', help='Keep only examples of these classes. Comma separated list.')  # noqa
@click.option('--only-images', help='Create dataset with specific examples. Useful to test model if your model has the ability to overfit.')  # noqa
@click.option('--limit-examples', type=int, help='Limit the dataset to the first `N` examples.')  # noqa
@click.option('--class-examples', type=int, help='Finish when every class has at least `N` number of samples. This will be the attempted lower bound; more examples might be added or a class might finish with fewer samples depending on the dataset.')  # noqa
@click.option('overrides', '--override', '-o', multiple=True, help='Custom parameters for readers.')  # noqa
@click.option('--debug', is_flag=True, help='Set level logging to DEBUG.')
def transform(dataset_reader, data_dir, output_dir, splits, only_classes,
              only_images, limit_examples, class_examples, overrides, debug):
    """
    Prepares dataset for ingestion.

    Converts the dataset into different (one per split) TFRecords files.
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)

    try:
        reader = get_reader(dataset_reader)
    except ValueError as e:
        tf.logging.error('Error getting reader: {}'.format(e))
        return

    # All splits must have a consistent set of classes.
    classes = None

    reader_kwargs = parse_override(overrides)

    try:
        for split in splits:
            # Create instance of reader.
            split_reader = reader(
                data_dir, split,
                only_classes=only_classes, only_images=only_images,
                limit_examples=limit_examples, class_examples=class_examples,
                **reader_kwargs
            )

            if classes is None:
                # "Save" classes from the first split reader
                classes = split_reader.classes
            else:
                # Overwrite classes after first split for consistency.
                split_reader.classes = classes

            # We assume we are saving object detection objects, but it should
            # be easy to modify once we have different types of objects.
            writer = ObjectDetectionWriter(split_reader, output_dir, split)
            writer.save()

            tf.logging.info('Composition per class ({}):'.format(split))
            for label, count in split_reader._per_class_counter.most_common():
                tf.logging.info(
                    '\t%s: %d', split_reader.pretty_name(label), count
                )

    except InvalidDataDirectory as e:
        tf.logging.error('Error reading dataset: {}'.format(e))
