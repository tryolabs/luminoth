import numpy as np

import click
import json
import os
import skvideo.io
import tensorflow as tf

from PIL import Image, ImageDraw
from luminoth.utils.predicting import PredictorNetwork


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

    # -- Initialize model --
    network = PredictorNetwork(config_files)

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
    tf.logging.info('Getting predictions for {} files'.format(total_files))

    #  -- Create output_dir if it doesn't exist --
    if output_dir:
        tf.gfile.MakeDirs(output_dir)

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
            prediction = network.predict_image(image)
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
            try:
                writer = skvideo.io.FFmpegWriter(save_path)
            except AssertionError as e:
                # from IPython import embed; embed(display_banner=False)
                tf.logging.error(e)
                tf.logging.error(
                    "Please install ffmpeg before making video predictions."
                )
                exit()
            num_of_frames = int(
                skvideo.io.ffprobe(file_path)['video']['@nb_frames'])
            video_progress_bar = click.progressbar(
                skvideo.io.vreader(file_path),
                length=num_of_frames,
                label='Predicting {}'.format(file_path))
            with video_progress_bar as bar:
                for frame in bar:
                    prediction = network.predict_image(frame)
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

        # Chose colors for bbox, the 60 and 255 correspond to transparency
        color = get_color(label)
        fill = tuple(color + [60])
        outline = tuple(color + [255])

        draw.rectangle(bbox, fill=fill, outline=outline)
        label = str(label)
        prob = '{:.2f}'.format(prob)
        draw.text(bbox[:2], '{} - {}'.format(label, prob))


def get_color(class_label):
    """Rudimentary way to create color palette for plotting clases

    Accepts integer or strings as class_labels
    """
    # We get these colors from the luminoth web client
    web_colors_hex = [
        'ff0029', '377eb8', '66a61e', '984ea3', '00d2d5', 'ff7f00', 'af8d00',
        '7f80cd', 'b3e900', 'c42e60', 'a65628', 'f781bf', '8dd3c7', 'bebada',
        'fb8072', '80b1d3', 'fdb462', 'fccde5', 'bc80bd', 'ffed6f', 'c4eaff',
        'cf8c00', '1b9e77', 'd95f02', 'e7298a', 'e6ab02', 'a6761d', '0097ff',
        '00d067', '000000', '252525', '525252', '737373', '969696', 'bdbdbd',
        'f43600', '4ba93b', '5779bb', '927acc', '97ee3f', 'bf3947', '9f5b00',
        'f48758', '8caed6', 'f2b94f', 'eff26e', 'e43872', 'd9b100', '9d7a00',
        '698cff', 'd9d9d9', '00d27e', 'd06800', '009f82', 'c49200', 'cbe8ff',
        'fecddf', 'c27eb6', '8cd2ce', 'c4b8d9', 'f883b0', 'a49100', 'f48800',
        '27d0df', 'a04a9b'
    ]
    hex_color = web_colors_hex[hash(class_label) % len(web_colors_hex)]
    return hex_to_rgb(hex_color)


def hex_to_rgb(x):
    return [int(x[i:i + 2], 16) for i in (0, 2, 4)]
