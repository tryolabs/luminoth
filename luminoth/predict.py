import numpy as np

import click
import json
import os
import skvideo.io
import tensorflow as tf

from PIL import Image, ImageDraw
from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.config import get_config, override_config_params
from luminoth.utils.predicting import PredictorNetwork

IMAGE_FORMATS = ['jpg', 'jpeg', 'png']
VIDEO_FORMATS = ['mov', 'mp4', 'avi']  # TODO: check if more formats work


def get_filetype(filename):
    extension = filename.split('.')[-1].lower()
    if extension in IMAGE_FORMATS:
        return 'image'
    elif extension in VIDEO_FORMATS:
        return 'video'


@click.command(help='Obtain a model\'s predictions on an image or directory of images.')  # noqa
@click.argument('path-or-dir')
@click.option('config_files', '--config', '-c', multiple=True, help='Config to use.')  # noqa
@click.option('--checkpoint', help='Checkpoint to use.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
@click.option('--output-dir', help='Where to write output')
@click.option('--save/--no-save', default=False, help='Save the image with the prediction of the model')  # noqa
@click.option('--min-prob', default=0.5, type=float, help='When drawing, only draw bounding boxes with probability larger than.')  # noqa
@click.option('--ignore-classes', default=None, multiple=True, help='Classes to ignore when predicting')  # noqa
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def predict(path_or_dir, config_files, checkpoint, override_params, output_dir,
            save, min_prob, ignore_classes, debug):
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Get file paths
    if tf.gfile.IsDirectory(path_or_dir):
        file_paths = [
            os.path.join(path_or_dir, f)
            for f in tf.gfile.ListDirectory(path_or_dir)
            if get_filetype(f) in ('image', 'video')
        ]
    else:
        if get_filetype(path_or_dir) in ('image', 'video'):
            file_paths = [path_or_dir]
        else:
            file_paths = []

    errors = 0
    successes = 0
    created_files_paths = []
    total_files = len(file_paths)
    if total_files == 0:
        no_files_message = ("No images or videos found. "
                            "Accepted formats -> Image: {} - Video: {}")
        tf.logging.error(no_files_message.format(IMAGE_FORMATS, VIDEO_FORMATS))
        exit()

    # Resolve the config to use and initialize the mdoel.
    if checkpoint:
        config = get_checkpoint_config(checkpoint)
    elif config_files:
        config = get_config(config_files)
    else:
        click.echo('You must specify either a checkpoint or a config file.')
        exit()

    if override_params:
        config = override_config_params(config, override_params)

    network = PredictorNetwork(config)

    # Create output_dir if it doesn't exist
    if output_dir:
        tf.gfile.MakeDirs(output_dir)

    tf.logging.info('Getting predictions for {} files'.format(total_files))

    # Iterate over file paths
    for file_path in file_paths:

        save_path = 'pred_' + os.path.basename(file_path)
        if output_dir:
            save_path = os.path.join(output_dir, save_path)

        if get_filetype(file_path) == 'image':
            click.echo('Predicting {}...'.format(file_path))
            with tf.gfile.Open(file_path, 'rb') as f:
                try:
                    image = Image.open(f).convert('RGB')
                except (tf.errors.OutOfRangeError, OSError) as e:
                    tf.logging.warning('Error: {}'.format(e))
                    tf.logging.warning("Couldn't open: {}".format(file_path))
                    errors += 1
                    continue

            # Run image through network
            prediction = network.predict_image(image)
            successes += 1

            # Filter results if required by user
            if ignore_classes:
                prediction = filter_classes(prediction, ignore_classes)

            # Save prediction json file
            with open(save_path + '.json', 'w') as outfile:
                json.dump(prediction, outfile)
            created_files_paths.append(save_path + '.json')

            # Save predicted image
            if save:
                with tf.gfile.Open(file_path, 'rb') as im_file:
                    image = Image.open(im_file)
                    draw_bboxes_on_image(image, prediction, min_prob)
                    image.save(save_path)
                created_files_paths.append(save_path)

        elif get_filetype(file_path) == 'video':
            # NOTE: We'll hardcode the video ouput to mp4 for the time being
            save_path = os.path.splitext(save_path)[0] + '.mp4'
            try:
                writer = skvideo.io.FFmpegWriter(save_path)
            except AssertionError as e:
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
                try:
                    for frame in bar:
                        # Run image through network
                        prediction = network.predict_image(frame)

                        # Filter results if required by user
                        if ignore_classes:
                            prediction = filter_classes(
                                prediction, ignore_classes)

                        image = Image.fromarray(frame)
                        draw_bboxes_on_image(image, prediction, min_prob)
                        writer.writeFrame(np.array(image))
                except RuntimeError as e:
                    click.echo()  # Error prints next to progress-bar if not
                    tf.logging.error('Error: {}'.format(e))
                    tf.logging.error('Corrupt videofile: {}'.format(file_path))
                    tf.logging.error(
                        'Partially processed video file saved in {}'.format(
                            save_path))
                    errors += 1

            writer.close()
            created_files_paths.append(save_path)

        else:
            tf.logging.warning("{} isn't an image/video".format(file_path))

    # Generate logs
    tf.logging.info(
        "Created the following files: {}".format(
            ', '.join(created_files_paths)
        )
    )

    if errors:
        tf.logging.warning('{} errors.'.format(errors))


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
        # label = str(label)
        color = get_color(label)
        fill = tuple(color + [60])
        outline = tuple(color + [255])

        draw.rectangle(bbox, fill=fill, outline=outline)
        prob = '{:.2f}'.format(prob)
        draw.text(bbox[:2], '{} - {}'.format(label, prob) if label else prob)


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


def filter_classes(prediction, ignore_classes):
    indexes_to_filter = [idx for idx, l in
                         enumerate(prediction['objects_labels'])
                         if l in ignore_classes]
    filtered_prediction = prediction.copy()

    # Replace some elements in filtered_prediction with their filtered version
    for key in ('objects', 'objects_labels', 'objects_labels_prob'):
        filtered_prediction[key] = [val for idx, val in
                                    enumerate(prediction[key])
                                    if idx not in indexes_to_filter]

    return filtered_prediction
