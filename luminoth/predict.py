import click
import json
import numpy as np
import os
import skvideo.io
import sys
import time
import tensorflow as tf
import uuid

from PIL import Image, ImageDraw
from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.config import get_config, override_config_params
from luminoth.utils.predicting import PredictorNetwork

from luminoth.utils.bbox_overlap import bbox_overlap

IMAGE_FORMATS = ['jpg', 'jpeg', 'png']
VIDEO_FORMATS = ['mov', 'mp4', 'avi']  # TODO: check if more formats work
FRAME_BUFFERLENGTH = 10
BUFFER_MATCH_THRESHOLD = 3
BUFFER_BBOX_OVERLAP = 0.1
MAX_OBJECT_AREA = 250 * 250
MIN_OBJECT_AREA = 20 * 20
NORM_OVERLAP = 0.3
NORM_DISTANCE = 80  # TODO: convert to a ratio and make proportional to frame size
NORM_EMB_DIST = 80
FPS = 6  # Can be None


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


def filter_by_size(objects, max_area=None, min_area=None):
    if max_area:
        objects = [o for o in objects if bbox_area(o['bbox']) < max_area]
    if min_area:
        objects = [o for o in objects if bbox_area(o['bbox']) > min_area]
    return objects


def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1] + 1)


def draw_bboxes_on_image(image, objects):
    # Open as 'RGBA' in order to draw translucent boxes.
    draw = ImageDraw.Draw(image, 'RGBA')

    for ind, obj in enumerate(objects):
        # Choose colors for bbox, the 60 and 255 correspond to transparency.
        if obj.get('tag'):
            color = get_color(str(obj['tag']))
            fill = tuple(color + [60])
            outline = tuple(color + [255])

            draw.rectangle(obj['bbox'], fill=fill, outline=outline)

            # # Draw the object's label.
            # prob = '{:.2f}'.format(obj['prob'])
            # label = '{} - {}'.format(obj['label'], prob) if obj['label'] else prob
            # feat_norm = np.linalg.norm(obj['feat'])

            # # -----PICK A LABEL------

            # 1) THIS LABEL SHOWS A COMPARISON OF RANDOM vs CONFUSED vs SAME embedding distances
            # label = "{:.1f} - {:.1f} - {:.1f}".format(
            #     obj['rand_dist'], obj['dist'], obj['min_dist']
            # ) if obj.get('dist') else str(feat_norm)

            # # 2) THIS LABEL SHOWS WHICH OBJECTS ARE CAUSING THE CONFUSION
            # label = "{} - {}".format(
            #     obj['tag'][:6], obj['confused_with'][:6]
            # ) if obj.get('confused_with') else "{}".format(obj['tag'][:6])

            # if obj.get('confused_with'):
            #     text_color = get_color(str(obj['confused_with']))
            #     text_fill = tuple(text_color + [60])
            # else:
            #     text_fill = fill

            # if obj.get('score'):
            #     label = '{:.2f} {:.2f} {:.2f}dsadasd {:.2f}'.format(obj['tag'][:4], obj['score'], obj['overlap'], obj['distance'], obj['emb_distance'])
            # else:
            #     label = str(obj['tag'][:4])

            label = 's:{:.2f}o:{:.2f}d:{:.2f}e:{:.2f}\n{}'.format(
                obj['debug']['score'], obj['debug']['overlap'],
                obj['debug']['distance'], obj['debug']['emb_distance'],
                obj['tag'][:3]
            )

            draw.text(obj['bbox'][:2], label)
        else:
            draw.rectangle(obj['bbox'])
            if obj.get('debug'):
                label = 's{:.2f} o{:.2f} d{:.2f} e{:.2f}'.format(
                    obj['debug']['score'], obj['debug']['overlap'],
                    obj['debug']['distance'], obj['debug']['emb_distance']
                )
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

    num_of_frames = int(skvideo.io.ffprobe(path)['video']['@nb_frames'])
    num, den = skvideo.io.ffprobe(path)['video']['@avg_frame_rate'].split('/')
    frame_rate = float(num) / float(den)
    print('frame_rate: ', frame_rate)
    frame_buffer_len = FRAME_BUFFERLENGTH if not FPS else (FRAME_BUFFERLENGTH * round((frame_rate / FPS)))

    video_progress_bar = click.progressbar(
        skvideo.io.vreader(path),
        length=num_of_frames,
        label='Predicting {}'.format(path)
    )

    objects_per_frame = []
    frame_buffer = []
    prediction_time = 0.0
    predictions = 0
    with video_progress_bar as bar:
        try:
            start_time = time.time()
            for idx, frame in enumerate(bar):
                # Run image through network.
                if not FPS or idx % int(round(frame_rate / FPS)) == 0:
                    start_pred_time = time.time()
                    objects = network.predict_image(frame)
                    stop_pred_time = time.time()
                    prediction_time += stop_pred_time - start_pred_time
                    predictions += 1

                    # Filter the results according to the user input.
                    objects = filter_classes(
                        objects,
                        only_classes=only_classes,
                        ignore_classes=ignore_classes
                    )

                    # Filter very large/very small objects.
                    objects = filter_by_size(
                        objects,
                        max_area=MAX_OBJECT_AREA,
                        min_area=MIN_OBJECT_AREA,
                    )

                    # Save detected objects in this frame
                    objects_per_frame.append({
                        'frame': idx,
                        'objects': objects
                    })

                    # Load initial buffer with the first frame_buffer_idx frames
                    # TODO: We should probably take this out of this loop and use a
                    #       separate loop for this that runs before this one.
                    if idx < frame_buffer_len:
                        frame_buffer.append(objects)
                        continue

                    # Provide tags to the current frame's objects
                    tag_objects(objects, frame_buffer)

                    # Update buffer
                    frame_buffer.append(objects)
                    frame_buffer.pop(0)

                # Draw the image and write it to the video file.
                image = Image.fromarray(frame)
                draw_bboxes_on_image(image, objects)
                writer.writeFrame(np.array(image))

                # # Update buffer
                # frame_buffer.append(objects)
                # frame_buffer.pop(0)

            stop_time = time.time()
            click.echo(
                'fps: {0:.1f}'.format(num_of_frames / (stop_time - start_time))
            )
            print('num_of_frames: ', num_of_frames)
            print('prediction time: ', prediction_time)
            print('predictions: ', predictions)
            print('total time: ', stop_time - start_time)

        except RuntimeError as e:
            click.echo()  # Error prints next to progress bar otherwise.
            click.echo('Error while processing {}: {}'.format(path, e))
            click.echo(
                'Partially processed video file saved in {}'.format(
                    save_path
                )
            )

    writer.close()

    return objects_per_frame


def tag_objects(current_frame_objects, frame_buffer):
    for idx, object_ in enumerate(current_frame_objects):
        object_match_chain = []
        last_matched_object = object_
        # NOTE: This is some really disgustin programming, fix!
        first_match = True
        first_frame = True

        for frame_objects in frame_buffer[::-1]:
            matching_object, debug_dict = get_matching_object(
                last_matched_object, frame_objects
            )
            if matching_object:
                # TODO: Check that the objects in the chain have the same tag
                object_match_chain.append(matching_object)
                last_matched_object = matching_object
                if first_match:
                    object_['debug'] = debug_dict
                    first_match = False
            elif debug_dict and first_frame:
                object_['debug'] = debug_dict
                first_frame = False
        # ========================================================
        # object_['tag'] = closest_last_frame_object['tag']

        # object_['dist'] = np.linalg.norm(
        #     closest_last_frame_object['feat'] - object_['feat']
        # )

        # object_['rand_dist'] = np.linalg.norm(
        #     random.choice(last_frame_objects)['feat'] - object_['feat']
        # )

        # distances = []
        # for lf_o in last_frame_objects:
        #     distances.append(
        #         np.linalg.norm(lf_o['feat'] - object_['feat'])
        #     )
        # min_dist_obj = last_frame_objects[np.argmin(distances)]
        # object_['min_dist'] = min(distances)
        # if object_['min_dist'] != object_['dist']:
        #     object_['confused_with'] = min_dist_obj['tag']
        # ========================================================

        if len(object_match_chain) > BUFFER_MATCH_THRESHOLD:
            if object_match_chain[0].get('tag'):
                object_['tag'] = object_match_chain[0]['tag']
            else:
                object_['tag'] = uuid.uuid4().hex


def get_matching_object(last_matched_object, frame_objects):
    # Filter objects of different class
    frame_objects = [
        o for o in frame_objects if o['label'] == last_matched_object['label']
    ]
    if not frame_objects:
        return None, None

    # Get bboxes
    object_bbox = np.array([last_matched_object['bbox']])
    frame_bboxes = np.array([object['bbox'] for object in frame_objects])

    # Match criteria
    bbox_overlaps = bbox_overlap(object_bbox, frame_bboxes)[0]
    bbox_distances = bbox_distance(object_bbox, frame_bboxes)
    object_embedding_distances = embedding_distance(
        last_matched_object, frame_objects
    )

    match_iter = zip(
        bbox_overlaps, bbox_distances, object_embedding_distances,
        frame_objects
    )
    # TODO: Vectorize
    best_score = 0
    for overlap, distance, embedding_dist, object_ in match_iter:
        # Match heuristic
        # normalized_distance = max(
        #     1 - (distance - NORM_DISTANCE) / NORM_DISTANCE, 0
        # )
        # normalized_emb_dist = max(
        #     1 - (embedding_dist - NORM_EMB_DIST) / NORM_EMB_DIST, 0
        # )
        normalized_distance = max((distance * (-2/NORM_DISTANCE)) + 1, 0)
        normalized_emb_dist = max((distance * (-2/NORM_EMB_DIST)) + 1, 0)
        score = 1.2 * overlap + normalized_distance + normalized_emb_dist

        if score > best_score:
            best_object = object_
            best_score = score

            debug_dict = {
                'score': score,
                'overlap': overlap,
                'distance': normalized_distance,
                'emb_distance': normalized_emb_dist,
            }

    if best_score > 1.7:
        return best_object, debug_dict
    elif best_score == 0:
        return None, None
    else:
        return None, debug_dict

def bbox_distance(bbox1, bboxes2):
    # Move to luminoth/utils and probably refactor
    bbox1 = bbox1[0]  # to match bbox_overlap interface 🤮
    centers_1 = np.vstack(
        [bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]]
    )
    centers_2 = np.vstack(
        [bboxes2[:, 2] - bboxes2[:, 0], bboxes2[:, 3] - bboxes2[:, 1]]
    )
    return np.linalg.norm(centers_1 - centers_2, axis=0)


def embedding_distance(last_matched_object, frame_objects):
    matched_object_embedding = last_matched_object['feat']
    frame_objects_embeddings = np.vstack([o['feat'] for o in frame_objects])
    return np.linalg.norm(
        matched_object_embedding - frame_objects_embeddings, axis=1
    )


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
