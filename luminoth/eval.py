import click
import json
import numpy as np
import os
import tensorflow as tf
import time

from luminoth.datasets import get_dataset
from luminoth.models import get_model
from luminoth.utils.bbox_overlap import bbox_overlap
from luminoth.utils.config import get_config
from luminoth.utils.image_vis import image_vis_summaries


@click.command(help='Evaluate trained (or training) models')
@click.option('dataset_split', '--split', default='val', help='Dataset split to use.')  # noqa
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Config to use.')  # noqa
@click.option('--watch/--no-watch', default=True, help='Keep watching checkpoint directory for new files.')  # noqa
@click.option('--from-global-step', type=int, default=None, help='Consider only checkpoints after this global step')  # noqa
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
@click.option('--files-per-class', type=int, default=10, help='How many files per class display in every epoch.')  # noqa
@click.option('--max-detections', type=int, default=100, help='Max detections to consider.')  # noqa
def eval(dataset_split, config_files, watch, from_global_step, override_params,
         files_per_class, max_detections):
    """Evaluate models using dataset."""

    # If the config file is empty, our config will be the base_config for the
    # default model.
    try:
        config = get_config(config_files, override_params=override_params)
    except KeyError:
        raise KeyError('model.type should be set on the custom config.')

    if not config.train.job_dir:
        raise KeyError('`job_dir` should be set.')
    if not config.train.run_name:
        raise KeyError('`run_name` should be set.')

    # `run_dir` is where the actual checkpoint and logs are located.
    run_dir = os.path.join(config.train.job_dir, config.train.run_name)

    # Only activate debug for if needed for debug visualization mode.
    if not config.train.debug:
        config.train.debug = config.eval.image_vis == 'debug'

    if config.train.debug or config.train.tf_debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Build the dataset tensors, overriding the default dataset split.
    config.dataset.split = dataset_split

    # Disable data augmentation.
    config.dataset.data_augmentation = []

    # Attempt to get class names, if available.
    classes_file = os.path.join(config.dataset.dir, 'classes.json')
    if tf.gfile.Exists(classes_file):
        class_labels = json.load(tf.gfile.GFile(classes_file))
    else:
        class_labels = None

    if config.model.type == 'fasterrcnn':
        # Override max detections with specified value.
        if config.model.network.with_rcnn:
            config.model.rcnn.proposals.total_max_detections = max_detections
        else:
            config.model.rpn.proposals.post_nms_top_n = max_detections

        # Also overwrite `min_prob_threshold` in order to use all detections.
        config.model.rcnn.proposals.min_prob_threshold = 0.0
    elif config.model.type == 'ssd':
        config.model.proposals.total_max_detections = max_detections
        config.model.proposals.min_prob_threshold = 0.0
    else:
        raise ValueError(
            "Model type '{}' not supported".format(config.model.type)
        )

    # Only a single run over the dataset to calculate metrics.
    config.train.num_epochs = 1

    # Seed setup.
    if config.train.seed:
        tf.set_random_seed(config.train.seed)

    # Set pretrained as not training.
    config.model.base_network.trainable = False

    model_class = get_model(config.model.type)
    model = model_class(config)
    dataset_class = get_dataset(config.dataset.type)
    dataset = dataset_class(config)
    train_dataset = dataset()

    train_image = train_dataset['image']
    train_objects = train_dataset['bboxes']
    train_filename = train_dataset['filename']

    # Build the graph of the model to evaluate, retrieving required
    # intermediate tensors.
    prediction_dict = model(train_image, train_objects)

    if config.model.type == 'ssd' or config.model.network.with_rcnn:
        pred = prediction_dict['classification_prediction']
        pred_objects = pred['objects']
        pred_objects_classes = pred['labels']
        pred_objects_scores = pred['probs']
    else:
        # Force the num_classes to 1.
        config.model.network.num_classes = 1

        pred = prediction_dict['rpn_prediction']
        pred_objects = pred['proposals']
        pred_objects_scores = pred['scores']
        # When using only RPN all classes are 0.
        pred_objects_classes = tf.zeros(
            (tf.shape(pred_objects_scores)[0],), dtype=tf.int32
        )

    # Retrieve *all* the losses from the model and calculate their streaming
    # means, so we get the loss over the whole dataset.
    batch_losses = model.loss(prediction_dict, return_all=True)
    losses = {}
    for loss_name, loss_tensor in batch_losses.items():
        loss_mean, _ = tf.metrics.mean(
            loss_tensor, name=loss_name,
            metrics_collections='metrics',
            updates_collections='metric_ops',
        )
        full_loss_name = '{}_losses/{}'.format(dataset_split, loss_name)
        losses[full_loss_name] = loss_mean

    metric_ops = tf.get_collection('metric_ops')

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    # Using a global saver instead of the one for the model.
    saver = tf.train.Saver(sharded=True, allow_empty=True)

    # Aggregate the required ops to evaluate into a dict.
    ops = {
        'init_op': init_op,
        'metric_ops': metric_ops,
        'pred_objects': pred_objects,
        'pred_objects_classes': pred_objects_classes,
        'pred_objects_scores': pred_objects_scores,
        'train_objects': train_objects,
        'losses': losses,
        'prediction_dict': prediction_dict,
        'filename': train_filename,
        'train_image': train_image
    }

    metrics_scope = '{}_metrics'.format(dataset_split)

    # Use global writer for all checkpoints. We don't want to write different
    # files for each checkpoint.
    writer = tf.summary.FileWriter(run_dir)

    files_to_visualize = {}

    last_global_step = from_global_step
    while True:
        # Get the checkpoint files to evaluate.
        try:
            checkpoints = get_checkpoints(
                run_dir, last_global_step, last_only=not watch
            )
        except ValueError as e:
            if not watch:
                tf.logging.error('Missing checkpoint.')
                raise e

            tf.logging.warning(
                'Missing checkpoint; Checking again in a moment')
            time.sleep(5)
            continue

        for checkpoint in checkpoints:
            # Always returned in order, so it's safe to assign directly.
            tf.logging.info(
                'Evaluating global_step {} using checkpoint \'{}\''.format(
                    checkpoint['global_step'], checkpoint['file']
                )
            )
            try:
                start = time.time()
                evaluate_once(
                    config, writer, saver, ops, checkpoint,
                    class_labels=class_labels,
                    metrics_scope=metrics_scope,
                    image_vis=config.eval.image_vis,
                    files_per_class=files_per_class,
                    files_to_visualize=files_to_visualize,
                )
                last_global_step = checkpoint['global_step']
                tf.logging.info('Evaluated in {:.2f}s'.format(
                    time.time() - start
                ))
            except tf.errors.NotFoundError:
                # The checkpoint is not ready yet. It was written in the
                # checkpoints file, but it still hasn't been completely saved.
                tf.logging.info(
                    'Checkpoint {} is not ready yet. '
                    'Checking again in a moment.'.format(
                        checkpoint['file']
                    )
                )
                time.sleep(5)
                continue

        # If no watching was requested, finish the execution.
        if not watch:
            return

        # Sleep for a moment and check for new checkpoints.
        tf.logging.info('All checkpoints evaluated; sleeping for a moment')
        time.sleep(5)


def get_checkpoints(run_dir, from_global_step=None, last_only=False):
    """Return all available checkpoints.

    Args:
        run_dir: Directory where the checkpoints are located.
        from_global_step (int): Only return checkpoints after this global step.
            The comparison is *strict*. If ``None``, returns all available
            checkpoints.

    Returns:
        List of dicts (with keys ``global_step``, ``file``) with all the
        checkpoints found.

    Raises:
        ValueError: If there are no checkpoints in ``run_dir``.
    """
    # The latest checkpoint file should be the last item of
    # `all_model_checkpoint_paths`, according to the CheckpointState protobuf
    # definition.
    # TODO: Must check if the checkpoints are complete somehow.
    ckpt = tf.train.get_checkpoint_state(run_dir)
    if not ckpt or not ckpt.all_model_checkpoint_paths:
        raise ValueError('Could not find checkpoint in {}.'.format(run_dir))

    # TODO: Any other way to get the global_step? (Same as in `checkpoints`.)
    checkpoints = sorted([
        {'global_step': int(path.split('-')[-1]), 'file': path}
        for path in ckpt.all_model_checkpoint_paths
    ], key=lambda c: c['global_step'])

    if last_only:
        checkpoints = checkpoints[-1:]
        tf.logging.info(
            'Using last checkpoint in run_dir, global_step = {}'.format(
                checkpoints[0]['global_step']
            )
        )
    elif from_global_step is not None:
        checkpoints = [
            c for c in checkpoints
            if c['global_step'] > from_global_step
        ]

        tf.logging.info(
            'Found %s checkpoints in run_dir with global_step > %s',
            len(checkpoints), from_global_step,
        )

    else:
        tf.logging.info(
            'Found {} checkpoints in run_dir'.format(len(checkpoints))
        )

    return checkpoints


def evaluate_once(config, writer, saver, ops, checkpoint,
                  class_labels, metrics_scope='metrics', image_vis=None,
                  files_per_class=None, files_to_visualize=None):
    """Run the evaluation once.

    Create a new session with the previously-built graph, run it through the
    dataset, calculate the evaluation metrics and write the corresponding
    summaries.

    Args:
        config: Config object for the model.
        writer: Summary writers.
        saver: Saver object to restore checkpoint parameters.
        ops (dict): All the operations needed to successfully run the model.
            Expects the following keys: ``init_op``, ``metric_ops``,
            ``pred_objects``, ``pred_objects_classes``,
            ``pred_objects_scores``, ``train_objects``, ``losses``,
            ``train_image``.
        checkpoint (dict): Checkpoint-related data.
            Expects the following keys: ``global_step``, ``file``.
    """
    # Output of the detector, per batch.
    output_per_batch = {
        'bboxes': [],  # Bounding boxes detected.
        'classes': [],  # Class associated to each bounding box.
        'scores': [],  # Score for each detection.
        'gt_bboxes': [],  # Ground-truth bounding boxes for the batch.
        'gt_classes': [],  # Ground-truth classes for each bounding box.
    }

    with tf.Session() as sess:
        sess.run(ops['init_op'])
        saver.restore(sess, checkpoint['file'])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        total_evaluated = 0
        start_time = time.time()

        try:
            track_start = start_time
            track_count = 0
            while not coord.should_stop():
                fetches = {
                    'metric_ops': ops['metric_ops'],
                    'bboxes': ops['pred_objects'],
                    'classes': ops['pred_objects_classes'],
                    'scores': ops['pred_objects_scores'],
                    'gt_bboxes': ops['train_objects'],
                    'losses': ops['losses'],
                    'filename': ops['filename'],
                }
                if image_vis is not None:
                    fetches['prediction_dict'] = ops['prediction_dict']
                    fetches['train_image'] = ops['train_image']

                batch_fetched = sess.run(fetches)
                output_per_batch['bboxes'].append(batch_fetched.get('bboxes'))
                output_per_batch['classes'].append(batch_fetched['classes'])
                output_per_batch['scores'].append(batch_fetched['scores'])

                batch_gt_objects = batch_fetched['gt_bboxes']
                output_per_batch['gt_bboxes'].append(batch_gt_objects[:, :4])
                batch_gt_classes = batch_gt_objects[:, 4]
                output_per_batch['gt_classes'].append(batch_gt_classes)

                val_losses = batch_fetched['losses']

                if image_vis is not None:
                    filename = batch_fetched['filename'].decode('utf-8')
                    visualize_file = False
                    for gt_class in batch_gt_classes:
                        cls_files = files_to_visualize.get(
                            gt_class, set()
                        )
                        if len(cls_files) < files_per_class:
                            files_to_visualize.setdefault(
                                gt_class, set()
                            ).add(filename)
                            visualize_file = True
                            break
                        elif filename in cls_files:
                            visualize_file = True
                            break

                    if visualize_file:
                        image_summaries = image_vis_summaries(
                            batch_fetched['prediction_dict'],
                            config=config.model,
                            extra_tag=filename,
                            image_visualization_mode=image_vis,
                            image=batch_fetched['train_image'],
                            gt_bboxes=batch_fetched['gt_bboxes']
                        )
                        for image_summary in image_summaries:
                            writer.add_summary(
                                image_summary, checkpoint['global_step']
                            )

                total_evaluated += 1
                track_count += 1

                track_end = time.time()
                if track_end - track_start > 20.:
                    click.echo(
                        '{} processed in {:.2f}s (global {:.2f} images/s, '
                        'period {:.2f} images/s)'.format(
                            total_evaluated, track_end - start_time,
                            total_evaluated / (track_end - start_time),
                            track_count / (track_end - track_start),
                        ))
                    track_count = 0
                    track_start = track_end

        except tf.errors.OutOfRangeError:

            # Save final evaluation stats into summary under the checkpoint's
            # global step.
            ap_per_class, ar_per_class = calculate_metrics(
                output_per_batch, config.model.network.num_classes
            )

            map_at_50 = np.mean(ap_per_class[:, 0])
            map_at_75 = np.mean(ap_per_class[:, 5])
            map_at_range = np.mean(ap_per_class)
            mar_at_range = np.mean(ar_per_class)

            tf.logging.info('Finished evaluation at step {}.'.format(
                checkpoint['global_step']))
            tf.logging.info('Evaluated {} images.'.format(total_evaluated))

            tf.logging.info(
                'Average Precision (AP) @ [0.50] = {:.3f}'.format(map_at_50)
            )
            tf.logging.info(
                'Average Precision (AP) @ [0.75] = {:.3f}'.format(map_at_75)
            )
            tf.logging.info(
                'Average Precision (AP) @ [0.50:0.95] = {:.3f}'.format(
                    map_at_range
                )
            )
            tf.logging.info(
                'Average Recall (AR) @ [0.50:0.95] = {:.3f}'.format(
                    mar_at_range
                )
            )

            for idx, val in enumerate(ap_per_class[:, 0]):
                class_label = '{} ({})'.format(
                    class_labels[idx], idx
                ) if class_labels else idx
                tf.logging.debug(
                    'Average Precision (AP) @ [0.50] for {} = {:.3f}'.format(
                        class_label, val
                    )
                )

            summary = [
                tf.Summary.Value(
                    tag='{}/AP@0.50'.format(metrics_scope),
                    simple_value=map_at_50
                ),
                tf.Summary.Value(
                    tag='{}/AP@0.75'.format(metrics_scope),
                    simple_value=map_at_75
                ),
                tf.Summary.Value(
                    tag='{}/AP@[0.50:0.95]'.format(metrics_scope),
                    simple_value=map_at_range
                ),
                tf.Summary.Value(
                    tag='{}/AR@[0.50:0.95]'.format(metrics_scope),
                    simple_value=mar_at_range
                ),
                tf.Summary.Value(
                    tag='{}/total_evaluated'.format(metrics_scope),
                    simple_value=total_evaluated
                ),
                tf.Summary.Value(
                    tag='{}/evaluation_time'.format(metrics_scope),
                    simple_value=time.time() - start_time
                ),
            ]

            for loss_name, loss_value in val_losses.items():
                tf.logging.debug('{} loss = {:.4f}'.format(
                    loss_name, loss_value))
                summary.append(tf.Summary.Value(
                    tag=loss_name,
                    simple_value=loss_value
                ))

            writer.add_summary(
                tf.Summary(value=summary), checkpoint['global_step']
            )

        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)


def calculate_metrics(output_per_batch, num_classes):
    """Calculates mAP and mAR from the detector's output.

    The procedure for calculating the average precision for class ``C`` is as
    follows (see `VOC mAP metric`_ for more details):

    Start by ranking all the predictions (for a given image and said class) in
    order of confidence.  Each of these predictions is marked as correct (true
    positive, when it has a IoU-threshold greater or equal to `iou_thresholds`)
    or incorrect (false positive, in the other case).  This matching is
    performed greedily over the confidence scores, so a higher-confidence
    prediction will be matched over another lower-confidence one even if the
    latter has better IoU.  Also, each prediction is matched at most once, so
    repeated detections are counted as false positives.

    We then integrate over the interpolated PR curve, thus obtaining the value
    for the class' average precision.  This interpolation makes sure the
    precision curve is monotonically decreasing; for this, we go through the
    precisions and make sure it's always decreasing.  The integration is
    performed over 101 fixed points over the curve (``[0.0, 0.01, ..., 1.0]``).

    Average the result among all the classes to obtain the final, ``mAP``,
    value.

    Args:
        output_per_batch (dict): Output of the detector to calculate mAP.
            Expects the following keys: ``bboxes``, ``classes``, ``scores``,
            ``gt_bboxes``, ``gt_classes``. Under each key, there should be a
            list of the results per batch as returned by the detector.
        num_classes (int): Number of classes on the dataset.

    Returns:
        (``np.ndarray``, ``ndarray``) tuple. The first value is an array of
        size (`num_classes`,), with the AP value per class, while the second
        one is an array for the AR.

    .. _VOC mAP metric:
        http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf
    """
    iou_thresholds = np.linspace(
        0.50, 0.95, np.round((0.95 - 0.50) / 0.05) + 1
    )
    # 101 recall levels, same as COCO evaluation.
    rec_thresholds = np.linspace(
        0.00, 1.00, np.round((1.00 - 0.00) / 0.01) + 1
    )

    # List; first by class, then by example. Each entry is a tuple of ndarrays
    # of size (D_{c,i},), for tp/fp labels and for score, where D_{c,i} is the
    # number of detected boxes for class `c` on image `i`.
    tp_fp_labels_by_class = [[] for _ in range(num_classes)]
    num_examples_per_class = [0 for _ in range(num_classes)]

    # For each image, order predictions by score and classify each as a true
    # positive or a false positive.
    num_batches = len(output_per_batch['bboxes'])
    for idx in range(num_batches):

        # Get the results of the batch.
        classes = output_per_batch['classes'][idx]  # (D_{c,i},)
        bboxes = output_per_batch['bboxes'][idx]  # (D_{c,i}, 4)
        scores = output_per_batch['scores'][idx]  # (D_{c,i},)

        gt_classes = output_per_batch['gt_classes'][idx]
        gt_bboxes = output_per_batch['gt_bboxes'][idx]

        # Analysis must be made per-class.
        for cls in range(num_classes):
            # Get the bounding boxes of `cls` only.
            cls_bboxes = bboxes[classes == cls, :]
            cls_scores = scores[classes == cls]
            cls_gt_bboxes = gt_bboxes[gt_classes == cls, :]

            num_gt = cls_gt_bboxes.shape[0]
            num_examples_per_class[cls] += num_gt

            # Sort by score descending, so we prioritize higher-confidence
            # results when matching.
            sorted_indices = np.argsort(-cls_scores)

            # Whether the ground-truth has been previously detected.
            is_detected = np.zeros((num_gt, len(iou_thresholds)))

            # TP/FP labels for detected bboxes of (class, image).
            tp_fp_labels = np.zeros((len(sorted_indices), len(iou_thresholds)))

            if num_gt == 0:
                # If no ground truth examples for class, all predictions must
                # be false positives.
                tp_fp_labels_by_class[cls].append(
                    (tp_fp_labels, cls_scores[sorted_indices])
                )
                continue

            # Get the IoUs for the class' bboxes.
            ious = bbox_overlap(cls_bboxes, cls_gt_bboxes)

            # Greedily assign bboxes to ground truths (highest score first).
            for bbox_idx in sorted_indices:
                gt_match = np.argmax(ious[bbox_idx, :])
                # TODO: Try to vectorize.
                for iou_idx, iou_threshold in enumerate(iou_thresholds):
                    if ious[bbox_idx, gt_match] >= iou_threshold:
                        # Over IoU threshold.
                        if not is_detected[gt_match, iou_idx]:
                            # And first detection: it's a true positive.
                            tp_fp_labels[bbox_idx, iou_idx] = True
                            is_detected[gt_match, iou_idx] = True

            tp_fp_labels_by_class[cls].append(
                (tp_fp_labels, cls_scores[sorted_indices])
            )

    # Calculate average precision per class.
    ap_per_class = np.zeros((num_classes, len(iou_thresholds)))
    ar_per_class = np.zeros((num_classes, len(iou_thresholds)))
    for cls in range(num_classes):
        tp_fp_labels = tp_fp_labels_by_class[cls]
        num_examples = num_examples_per_class[cls]

        # Flatten the tp/fp labels into a single ndarray.
        labels, scores = zip(*tp_fp_labels)
        labels = np.concatenate(labels)
        scores = np.concatenate(scores)

        # Sort the tp/fp labels by decreasing confidence score and calculate
        # precision and recall at every position of this ranked output.
        sorted_indices = np.argsort(-scores)
        true_positives = labels[sorted_indices, :]
        false_positives = 1 - true_positives

        cum_true_positives = np.cumsum(true_positives, axis=0)
        cum_false_positives = np.cumsum(false_positives, axis=0)

        recall = cum_true_positives.astype(float) / num_examples
        precision = np.divide(
            cum_true_positives.astype(float),
            cum_true_positives + cum_false_positives
        )

        # Find AP by integrating over PR curve, with interpolated precision.
        for iou_idx in range(len(iou_thresholds)):
            p = precision[:, iou_idx]
            r = recall[:, iou_idx]

            # Interpolate the precision. (Make it monotonically-increasing.)
            for i in range(len(p) - 1, 0, -1):
                if p[i] > p[i-1]:
                    p[i-1] = p[i]

            ap = 0
            inds = np.searchsorted(r, rec_thresholds)
            for ridx, pidx in enumerate(inds):
                if pidx >= len(r):
                    # Out of bounds, no recall higher than threshold for any of
                    # the remaining thresholds (as they're ordered).
                    break

                ap += p[pidx] / len(rec_thresholds)

            ap_per_class[cls, iou_idx] = ap
            if len(r):
                ar_per_class[cls, iou_idx] = r[-1]
            else:
                ar_per_class[cls, iou_idx] = 0

    return ap_per_class, ar_per_class


if __name__ == '__main__':
    eval()
