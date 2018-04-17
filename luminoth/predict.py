import click
import json
import numpy as np
import os
import skvideo.io
import sys
import time
import tensorflow as tf

from PIL import Image, ImageDraw
from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.config import get_config, override_config_params
from luminoth.utils.predicting import PredictorNetwork

IMAGE_FORMATS = ['jpg', 'jpeg', 'png']
VIDEO_FORMATS = ['mov', 'mp4', 'avi']  # TODO: check if more formats work


def get_file_type(filename):
    extension = filename.split('.')[-1].lower()
    if extension in IMAGE_FORMATS:
        return 'image'
    elif extension in VIDEO_FORMATS:
        return 'video'


def resolve_files(path_or_dir):
    """Returns the file paths for `path_or_dir`.

    Args:
        path_or_dir: String or list of strings for the paths or directories to
            run predictions in. For directories, will return all the files
            within.

    Returns:
        List of strings with the full path for each file.
    """
    if not isinstance(path_or_dir, tuple):
        path_or_dir = (path_or_dir,)

    paths = []
    for entry in path_or_dir:
        if tf.gfile.IsDirectory(entry):
            paths.extend([
                os.path.join(entry, f)
                for f in tf.gfile.ListDirectory(entry)
                if get_file_type(f) in ('image', 'video')
            ])
        elif get_file_type(entry) in ('image', 'video'):
            if not tf.gfile.Exists(entry):
                click.echo('Input {} not found, skipping.'.format(entry))
                continue
            paths.append(entry)

    return paths


def filter_classes(objects, only_classes=None, ignore_classes=None):
    if ignore_classes:
        objects = [o for o in objects if o['label'] not in ignore_classes]

    if only_classes:
        objects = [o for o in objects if o['label'] in only_classes]

    return objects


def draw_bboxes_on_image(image, objects):
    # Open as 'RGBA' in order to draw translucent boxes.
    draw = ImageDraw.Draw(image, 'RGBA')

    for ind, obj in enumerate(objects):
        # Choose colors for bbox, the 60 and 255 correspond to transparency.
        color = get_color(str(obj['label']))
        fill = tuple(color + [60])
        outline = tuple(color + [255])

        draw.rectangle(obj['bbox'], fill=fill, outline=outline)

        # Draw the object's label.
        prob = '{:.2f}'.format(obj['prob'])
        label = '{} - {}'.format(obj['label'], prob) if obj['label'] else prob
        draw.text(obj['bbox'][:2], label)


def get_color(class_label):
    """Rudimentary way to create color palette for plotting clases.

    Accepts integer or strings as class_labels.
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


def predict_image(network, path, only_classes=None, ignore_classes=None,
                  save_path=None):
    click.echo('Predicting {}...'.format(path), nl=False)

    # Open and read the image to predict.
    with tf.gfile.Open(path, 'rb') as f:
        try:
            image = Image.open(f).convert('RGB')
        except (tf.errors.OutOfRangeError, OSError) as e:
            click.echo()
            click.echo('Error while processing {}: {}'.format(path, e))
            return

    # Run image through the network.
    objects = network.predict_image(image)

    # Filter the results according to the user input.
    objects = filter_classes(
        objects,
        only_classes=only_classes,
        ignore_classes=ignore_classes
    )

    # Save predicted image.
    if save_path:
        draw_bboxes_on_image(image, objects)
        image.save(save_path)

    click.echo(' done.')
    return objects


def predict_video(network, path, only_classes=None, ignore_classes=None,
                  save_path=None):
    if save_path:
        # We hardcode the video ouput to mp4 for the time being.
        save_path = os.path.splitext(save_path)[0] + '.mp4'
        try:
            writer = skvideo.io.FFmpegWriter(save_path)
        except AssertionError as e:
            tf.logging.error(e)
            tf.logging.error(
                'Please install ffmpeg before making video predictions.'
            )
            exit()
    else:
        click.echo(
            'Video not being saved. Note that for the time being, no JSON '
            'output is being generated. Did you mean to specify `--save-path`?'
        )

    num_of_frames = int(skvideo.io.ffprobe(path)['video']['@nb_frames'])

    video_progress_bar = click.progressbar(
        skvideo.io.vreader(path),
        length=num_of_frames,
        label='Predicting {}'.format(path)
    )

    objects_per_frame = []
    with video_progress_bar as bar:
        try:
            start_time = time.time()
            for idx, frame in enumerate(bar):
                # Run image through network.
                objects = network.predict_image(frame)

                # Filter the results according to the user input.
                objects = filter_classes(
                    objects,
                    only_classes=only_classes,
                    ignore_classes=ignore_classes
                )

                objects_per_frame.append({
                    'frame': idx,
                    'objects': objects
                })

                # Draw the image and write it to the video file.
                image = Image.fromarray(frame)
                draw_bboxes_on_image(image, objects)
                if save_path:
                    writer.writeFrame(np.array(image))

            stop_time = time.time()
            click.echo(
                'fps: {0:.1f}'.format(num_of_frames / (stop_time - start_time))
            )
        except RuntimeError as e:
            click.echo()  # Error prints next to progress bar otherwise.
            click.echo('Error while processing {}: {}'.format(path, e))
            if save_path:
                click.echo(
                    'Partially processed video file saved in {}'.format(
                        save_path
                    )
                )

    if save_path:
        writer.close()

    return objects_per_frame


@click.command(help="Obtain a model's predictions.")
@click.argument('path-or-dir', nargs=-1)
@click.option('config_files', '--config', '-c', multiple=True, help='Config to use.')  # noqa
@click.option('--checkpoint', help='Checkpoint to use.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
@click.option('output_path', '--output', '-f', default='-', help='Output file.')  # noqa
@click.option('--save-media-to', '-d', help='Directory to store media to.')
@click.option('--min-prob', default=0.5, type=float, help='When drawing, only draw bounding boxes with probability larger than.')  # noqa
@click.option('--max-detections', default=100, type=int, help='Maximum number of detections per image.')  # noqa
@click.option('--only-class', '-k', default=None, multiple=True, help='Class to ignore when predicting.')  # noqa
@click.option('--ignore-class', '-K', default=None, multiple=True, help='Class to ignore when predicting.')  # noqa
@click.option('--debug', is_flag=True, help='Set debug level logging.')
def predict(path_or_dir, config_files, checkpoint, override_params,
            output_path, save_media_to, min_prob, max_detections, only_class,
            ignore_class, debug):
    """Obtain a model's predictions.

    Receives either `config_files` or `checkpoint` in order to load the correct
    model. Afterwards, runs the model through the inputs specified by
    `path-or-dir`, returning predictions according to the format specified by
    `output`.

    Additional model behavior may be modified with `min-prob`, `only-class` and
    `ignore-class`.
    """
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)

    if only_class and ignore_class:
        click.echo(
            "Only one of `only-class` or `ignore-class` may be specified."
        )
        return

    # Process the input and get the actual files to predict.
    files = resolve_files(path_or_dir)
    if not files:
        error = 'No files to predict found. Accepted formats are: {}.'.format(
            ', '.join(IMAGE_FORMATS + VIDEO_FORMATS)
        )
        click.echo(error)
        return
    else:
        click.echo('Found {} files to predict.'.format(len(files)))

    # Build the `Formatter` based on the outputs, which automatically writes
    # the formatted output to all the requested output files.
    if output_path == '-':
        output = sys.stdout
    else:
        output = open(output_path, 'w')

    # Create `save_media_to` if specified and it doesn't exist.
    if save_media_to:
        tf.gfile.MakeDirs(save_media_to)

    # Resolve the config to use and initialize the model.
    if checkpoint:
        config = get_checkpoint_config(checkpoint)
    elif config_files:
        config = get_config(config_files)
    else:
        click.echo(
            'Neither checkpoint not config specified, assuming `accurate`.'
        )
        config = get_checkpoint_config('accurate')

    if override_params:
        config = override_config_params(config, override_params)

    # Filter bounding boxes according to `min_prob` and `max_detections`.
    if config.model.type == 'fasterrcnn':
        if config.model.network.with_rcnn:
            config.model.rcnn.proposals.total_max_detections = max_detections
        else:
            config.model.rpn.proposals.post_nms_top_n = max_detections
        config.model.rcnn.proposals.min_prob_threshold = min_prob
    elif config.model.type == 'ssd':
        config.model.proposals.total_max_detections = max_detections
        config.model.proposals.min_prob_threshold = min_prob
    else:
        raise ValueError(
            "Model type '{}' not supported".format(config.model.type)
        )

    # Instantiate the model indicated by the config.
    network = PredictorNetwork(config)

    # Iterate over files and run the model on each.
    for file in files:

        # Get the media output path, if media storage is requested.
        save_path = os.path.join(
            save_media_to, 'pred_{}'.format(os.path.basename(file))
        ) if save_media_to else None

        file_type = get_file_type(file)
        predictor = predict_image if file_type == 'image' else predict_video

        objects = predictor(
            network, file,
            only_classes=only_class,
            ignore_classes=ignore_class,
            save_path=save_path,
        )

        # TODO: Not writing jsons for video files for now.
        if objects is not None and file_type == 'image':
            output.write(
                json.dumps({
                    'file': file,
                    'objects': objects,
                }) + '\n'
            )

    output.close()
