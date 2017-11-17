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
@click.argument('path-or-dir')
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Config to use.')  # noqa
@click.option('--output-dir', help='Where to write output')
@click.option('--save/--no-save', default=False, help='Save the image with the prediction of the model')  # noqa
@click.option('--min-prob', default=0.5, type=float, help='When drawing, only draw bounding boxes with probability larger than.')  # noqa
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def predict(path_or_dir, config_files, output_dir, save, min_prob, debug):
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    multiple = False
    if tf.gfile.IsDirectory(path_or_dir):
        image_paths = [
            os.path.join(path_or_dir, f)
            for f in tf.gfile.ListDirectory(path_or_dir)
            if is_image(f)
        ]
        multiple = True
    else:
        image_paths = [path_or_dir]

    total_images = len(image_paths)
    results = []
    errors = []

    tf.logging.info('Getting predictions for {} files.'.format(total_images))

    prediction_iter = get_predictions(image_paths, config_files)
    if multiple:
        with click.progressbar(prediction_iter, length=total_images) as preds:
            for prediction in preds:
                if prediction.get('error') is None:
                    results.append(prediction)
                else:
                    errors.append(prediction)
    else:
        for prediction in prediction_iter:
            if prediction.get('error') is None:
                results.append(prediction)
            else:
                errors.append(prediction)

    if multiple:
        tf.logging.info('{} images with predictions'.format(len(results)))
    else:
        tf.logging.info(
            '{} objects detected'.format(len(results[0]['objects'])))

    if len(errors):
        tf.logging.warning('{} errors.'.format(len(errors)))

    dir_log = output_dir if output_dir else 'current directory'
    if save:
        tf.logging.info(
            'Saving results and images with bounding boxes drawn in {}'.format(
                dir_log))
    else:
        tf.logging.info('Saving results in {}'.format(dir_log))

    if output_dir:
        # Create dir if it doesn't exists
        tf.gfile.MakeDirs(output_dir)

    for res in results:
        image_path = res['image_path']
        save_path = 'pred_' + os.path.basename(image_path)
        if output_dir:
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
            image.save(save_path)
