import click
import os
import tensorflow as tf

from PIL import Image

from luminoth.utils.dataset import (
    read_xml, read_image, to_int64, to_string, to_bytes
)
from .dataset import DatasetTool, RecordSaver


def adjust_bbox(xmin, ymin, xmax, ymax, old_width, old_height,
                new_width, new_height):
    # TODO: consider reusing luminoth.utils.image.adjust_bboxes instead of
    # this, but note it uses tensorflow, and using tf and np here may introduce
    # too many problems.
    xmin = (xmin / old_width) * new_width
    ymin = (ymin / old_height) * new_height
    xmax = (xmax / old_width) * new_width
    ymax = (ymax / old_height) * new_height

    return xmin, ymin, xmax, ymax


class ImageNet(DatasetTool):
    def __init__(self, data_dir):
        super(ImageNet, self).__init__()
        self._data_dir = data_dir

    def read_classes(self):
        path = os.path.join(
            self._data_dir, 'Annotations', 'DET', 'train', 'ILSVRC2013_train'
        )
        classes = set()
        for entry in tf.gfile.ListDirectory(path):
            classes.add(entry)

        return list(sorted(classes))

    def load_split(self, split='train'):
        if split not in self.VALID_SPLITS:
            raise ValueError

        split_path = os.path.join(
            self._data_dir, 'ImageSets', 'DET', '{}.txt'.format(split)
        )
        with tf.gfile.GFile(split_path) as f:
            for line in f:
                # The images in 'extra' directories don't have annotations.
                if 'extra' in line:
                    continue
                filename = line.split()[0]
                filename = os.path.join(split, filename)
                yield filename.strip()

    def get_image_path(self, image_id):
        return os.path.join(
            self._data_dir, 'Data', 'DET', '{}.JPEG'.format(image_id)
        )

    def get_image_annotation(self, image_id):
        return os.path.join(
            self._data_dir, 'Annotations', 'DET', '{}.xml'.format(image_id)
        )

    def image_to_example(self, classes, image_id):
        annotation_path = self.get_image_annotation(image_id)
        image_path = self.get_image_path(image_id)

        # Read both the image and the annotation into memory.
        annotation = read_xml(annotation_path)
        image = read_image(image_path)

        # TODO: consider alternatives to using Pillow here.
        image_pil = Image.open(image_path)
        width = image_pil.width
        height = image_pil.height
        image_pil.close()

        obj_vals = {
            'label': [],
            'xmin': [],
            'ymin': [],
            'xmax': [],
            'ymax': [],
        }

        objects = annotation.get('object')
        if objects is None:
            # If there's no bounding boxes, we don't want it
            return
        for b in annotation['object']:
            try:
                label_id = classes.index(b['name'])
            except ValueError:
                continue

            (xmin, ymin, xmax, ymax) = adjust_bbox(
                xmin=int(b['bndbox']['xmin']), ymin=int(b['bndbox']['ymin']),
                xmax=int(b['bndbox']['xmax']), ymax=int(b['bndbox']['ymax']),
                old_width=int(annotation['size']['width']),
                old_height=int(annotation['size']['height']),
                new_width=width, new_height=height
            )
            obj_vals['label'].append(to_int64(label_id))
            obj_vals['xmin'].append(to_int64(xmin))
            obj_vals['ymin'].append(to_int64(ymin))
            obj_vals['xmax'].append(to_int64(xmax))
            obj_vals['ymax'].append(to_int64(ymax))

        if len(obj_vals['label']) == 0:
            # No bounding box matches the available classes.
            return

        object_feature_lists = {
            'label': tf.train.FeatureList(feature=obj_vals['label']),
            'xmin': tf.train.FeatureList(feature=obj_vals['xmin']),
            'ymin': tf.train.FeatureList(feature=obj_vals['ymin']),
            'xmax': tf.train.FeatureList(feature=obj_vals['xmax']),
            'ymax': tf.train.FeatureList(feature=obj_vals['ymax']),
        }

        object_features = tf.train.FeatureLists(
            feature_list=object_feature_lists
        )

        sample = {
            'width': to_int64(width),
            'height': to_int64(height),
            'depth': to_int64(3),
            'filename': to_string(annotation['filename']),
            'image_raw': to_bytes(image),
        }

        # Now build an `Example` protobuf object and save with the writer.
        context = tf.train.Features(feature=sample)
        example = tf.train.SequenceExample(
            feature_lists=object_features, context=context
        )

        return example


@click.command()
@click.option('--data-dir', default='datasets/imagenet')
@click.option('--output-dir', default='datasets/imagenet/tf')
@click.option('ignore_splits', '--ignore-split', multiple=True)
@click.option('--only-filename', help='Create dataset with a single example.')
@click.option('--limit-examples', type=int, help='Limit dataset with to the first `N` examples.')  # noqa
@click.option('--limit-classes', type=int, help='Limit dataset with `N` random classes.')  # noqa
@click.option('--seed', type=int, help='Seed used for picking random classes.')
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def imagenet(data_dir, output_dir, ignore_splits, only_filename,
             limit_examples, limit_classes, seed, debug):
    """
    Prepares ImageNet dataset for ingestion.
    """
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    imagenet = ImageNet(data_dir=data_dir)
    saver = RecordSaver(
        imagenet, output_dir,
        ignore_splits=ignore_splits,
        only_filename=only_filename,
        limit_examples=limit_examples,
        limit_classes=limit_classes,
        seed=seed
    )

    saver.save()
