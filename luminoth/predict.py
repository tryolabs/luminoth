import numpy as np

import click
import json
import os
import skvideo.io
import tensorflow as tf

from PIL import Image, ImageDraw

from luminoth.utils.predicting import network_gen


def is_image(filename):
    f = filename.lower()
    return f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')


def is_video(filename):
    f = filename.lower()
    # TODO: check more video formats
    return f.endswith('.mov') or f.endswith('.mp4')


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

    # -- Get file paths --
    if tf.gfile.IsDirectory(path_or_dir):
        file_paths = [
            os.path.join(path_or_dir, f)
            for f in tf.gfile.ListDirectory(path_or_dir)
            if is_image(f) or is_video(f)
        ]
    else:
        file_paths = [path_or_dir]

    errors = 0
    successes = 0
    total_files = len(file_paths)
    tf.logging.info('Getting predictions for {} files.'.format(total_files))

    #  -- Create output_dir if it doesn't exist --
    if output_dir:
        tf.gfile.MakeDirs(output_dir)

    # -- Initialize model --
    network_iter = network_gen(config_files)
    next(network_iter)

    # -- Iterate over file paths --
    for file_path in file_paths:

        save_path = 'pred_' + os.path.basename(file_path)
        if output_dir:
            save_path = os.path.join(output_dir, save_path)

        if is_image(file_path):
            print('Predicting {}...'.format(file_path))
            with tf.gfile.Open(file_path, 'rb') as f:
                try:
                    image = Image.open(f).convert('RGB')
                except tf.errors.OutOfRangeError as e:
                    tf.logging.warning('Error: {}'.format(e))
                    tf.logging.warning('{} failed.'.format(file_path))
                    errors += 1
                    continue

            # Run image through network
            prediction = network_iter.send(image)
            successes += 1

            # -- Save results --
            with open(save_path + '.json', 'w') as outfile:
                json.dump(prediction, outfile)
            if save:
                with tf.gfile.Open(file_path, 'rb') as im_file:
                    image = Image.open(im_file)
                    draw_bboxes_on_image(image, prediction, min_prob)
                    image.save(save_path)

        elif is_video(file_path):
            # We'll hardcode the video ouput to mp4 for now
            save_path = os.path.splitext(save_path)[0] + '.mp4'
            writer = skvideo.io.FFmpegWriter(save_path)
            num_of_frames = int(
                skvideo.io.ffprobe(file_path)['video']['@nb_frames'])
            video_progress_bar = click.progressbar(
                skvideo.io.vreader(file_path),
                length=num_of_frames,
                label='Predicting {}'.format(file_path))
            with video_progress_bar as bar:
                for frame in bar:
                    prediction = network_iter.send(frame)
                    image = Image.fromarray(frame)
                    draw_bboxes_on_image(image, prediction, min_prob)
                    writer.writeFrame(np.array(image))
                writer.close()

        else:
            tf.logging.warning("{} isn't an image/video".format(file_path))

    # -- Generate logs --
    logs_dir = output_dir if output_dir else 'current directory'
    saving_logs_message = ('Saving results and tagged images/videos in {}'
                           if save else 'Saving results in {}')
    tf.logging.info(saving_logs_message.format(logs_dir))

    if errors:
        tf.logging.warning('{} errors.'.format(errors))

    if len(file_paths) > 1:
        tf.logging.info('Predicted {} files'.format(successes))
    else:
        tf.logging.info(
            '{} objects detected'.format(len(prediction['objects']))
        )


def draw_bboxes_on_image(image, prediction, min_prob):
    # Open as 'RGBA' in order to draw translucent boxes
    draw = ImageDraw.Draw(image, 'RGBA')

    objects = prediction['objects']
    labels = prediction['objects_labels']
    probs = prediction['objects_labels_prob']
    object_iter = zip(objects, labels, probs)

    for ind, (bbox, label, prob) in enumerate(object_iter):
        if prob < min_prob:
            continue

        draw.rectangle(bbox, fill=(0, 0, 255, 60), outline=(0, 0, 255, 150))
        label = str(label)
        prob = '{:.2f}'.format(prob)
        draw.text(bbox[:2], '{} - {}'.format(label, prob))
