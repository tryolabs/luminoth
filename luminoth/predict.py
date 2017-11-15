import click
import json
import os
import tensorflow as tf

from PIL import Image, ImageDraw

from luminoth.utils.predicting import get_predictions


def is_image(filename):
    f = filename.lower()
    return f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')


@click.command(help='Obtain a model\'s predictions on an image or directory of images.')  # noqa
@click.argument('image-path')
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Config to use.')  # noqa
@click.option('--output-dir', help='Where to write output')
@click.option('--save/--no-save', default=False, help='Save the image with the prediction of the model')  # noqa
@click.option('--min-prob', default=0.5, type=float, help='When drawing, only draw bounding boxes with probability larger than.')  # noqa
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def predict(image_path, config_files, output_dir, save, min_prob, debug):
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    multiple = False
    if tf.gfile.IsDirectory(image_path):
        image_paths = [
            os.path.join(image_path, f)
            for f in tf.gfile.ListDirectory(image_path)
            if is_image(f)
        ]
        multiple = True
    else:
        image_paths = [image_path]

    results = get_predictions(image_paths, config_files)
    errors = [r for r in results if r.get('error') is not None]
    results = [r for r in results if r.get('error') is None]

    if multiple:
        tf.logging.info('{} images with predictions'.format(len(results)))
    else:
        tf.logging.info(
            '{} objects detected'.format(len(results[0]['objects'])))

    if len(errors):
        tf.logging.warning('{} errors.'.format(len(errors)))

    for res in results:
        image_path = res['image_path']
        save_path = 'pred_' + os.path.basename(image_path)
        if output_dir:
            # Create dir if it doesn't exists
            tf.gfile.MakeDirs(output_dir)
            save_path = os.path.join(output_dir, save_path)

        with open(save_path + '.json', 'w') as outfile:
            json.dump(res, outfile)

        if save:
            with tf.gfile.Open(image_path, 'rb') as im_file:
                image = Image.open(im_file)
            # Draw bounding boxes
            draw = ImageDraw.Draw(image)

            objects = res['objects']
            labels = res['objects_labels']
            probs = res['objects_labels_prob']
            object_iter = zip(objects, labels, probs)

            for ind, (bbox, label, prob) in enumerate(object_iter):
                if prob < min_prob:
                    continue

                draw.rectangle(bbox, outline='red')
                label = str(label)
                prob = '{:.2f}'.format(prob)
                draw.text(bbox[:2], '{} - {}'.format(label, prob))

            # Save the image
            tf.logging.info(
                'Saving image with bounding boxes in {}'.format(save_path))
            image.save(save_path)
