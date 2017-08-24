import click
import numpy as np
import os
import tensorflow as tf
import time

from .dataset import TFRecordDataset
from .models import MODELS, PRETRAINED_MODELS
from .utils.config import (
    load_config, merge_into, parse_override
)
from .utils.vars import get_saver
from .utils.bbox import bbox_overlaps


@click.command(help='Evaluate trained (or training) models')
@click.argument('model-type', type=click.Choice(MODELS.keys()))
@click.argument('dataset-split', default='val')
@click.option('config_file', '--config', '-c', type=click.File('r'), help='Config to use.')
@click.option('--model-dir', required=True, help='Directory from where to read saved models.')
@click.option('--log-dir', help='Directory where to save evaluation logs.')
@click.option('--all-checkpoints', is_flag=True, default=False, help='Whether to evaluate all or last checkpoint.')
@click.option('--watch/--no-watch', default=True, help='Keep watching checkpoint directory for new files.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')
def evaluate(model_type, dataset_split, config_file, model_dir, log_dir,
             all_checkpoints, watch, override_params):
    """
    Evaluate models using dataset.
    """
    model_class = MODELS[model_type.lower()]
    config = model_class.base_config

    if config_file:
        # If we have a custom config file overwritting default settings
        # then we merge those values to the base_config.
        custom_config = load_config(config_file)
        config = merge_into(custom_config, config)

    config.train.model_dir = model_dir or config.train.model_dir
    config.train.log_dir = log_dir or config.train.log_dir

    if override_params:
        override_config = parse_override(override_params)
        config = merge_into(override_config, config)

    # Build the dataset tensors, overriding the default dataset split.
    config.dataset.split = dataset_split

    # Only a single run over the dataset to calculate metrics.
    config.train.num_epochs = 1

    model = model_class(config)
    pretrained = PRETRAINED_MODELS[config.pretrained.net](
        config.pretrained
    )
    dataset = TFRecordDataset(config)
    train_dataset = dataset()

    train_image = train_dataset['image']
    train_filename = train_dataset['filename']
    train_objects = train_dataset['bboxes']

    # TODO: This is not the best place to configure rank? Why is rank not
    # transmitted through the queue
    train_image.set_shape((None, None, 3))
    # We add fake batch dimension to train data. TODO: DEFINITELY NOT THE BEST
    # PLACE
    train_image = tf.expand_dims(train_image, 0)

    # Build the graph of the model to evaluate, retrieving required
    # intermediate tensors.
    pretrained_dict = pretrained(train_image, is_training=False)
    prediction_dict = model(
        train_image, pretrained_dict['net'], train_objects, is_training=False
    )

    pred = prediction_dict['classification_prediction']
    pred_objects = pred['objects']
    pred_objects_classes = pred['objects_labels']
    pred_objects_scores = pred['objects_labels_prob']

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
        full_loss_name = 'val_losses/{}'.format(loss_name)
        losses[full_loss_name] = loss_mean

    metric_ops = tf.get_collection('metric_ops')

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    # Get the saver required to load model parameters.
    saver = get_saver((model, pretrained, ))

    # Aggregate the required ops to evaluate into a dict..
    ops = {
        'init_op': init_op,
        'metric_ops': metric_ops,
        'pred_objects': pred_objects,
        'pred_objects_classes': pred_objects_classes,
        'pred_objects_scores': pred_objects_scores,
        'train_filename': train_filename,
        'train_objects': train_objects,
        'losses': losses,
    }

    last_global_step = None
    while True:
        # Get the checkpoint files to evaluate.
        checkpoints = get_checkpoints(config, last_global_step)

        # TODO: Change parameter to `from_global_step`.
        # We only want to filter on the first iteration.
        if last_global_step is not None and not all_checkpoints:
            checkpoints = [checkpoints[-1]] if checkpoints else []

        for checkpoint in checkpoints:
            # Always returned in order, so it's safe to assign directly.
            tf.logging.info(
                'Evaluating global_step %s using checkpoint \'%s\'',
                checkpoint['global_step'], checkpoint['file']
            )
            last_global_step = checkpoint['global_step']
            evaluate_once(config, saver, ops, checkpoint)

        # If no watching was requested, finish the execution.
        if not watch:
            return

        # Sleep for a minute and check for new checkpoints.
        time.sleep(5 * 60)


def get_checkpoints(config, from_global_step=None):
    """Return all available checkpoints.

    Args:
        config: Run configuration file, where the checkpoint dir is present.
        from_global_step (int): Only return checkpoints after this global step.
            The comparison is *strict*. If ``None``, returns all available
            checkpoints.

    Returns:
        List of dicts (with keys ``global_step``, ``file``) with all the
        checkpoints found.

    Raises:
        ValueError: If there are no checkpoints on the ``train.model_dir`` key
            of `config`.
    """
    # The latest checkpoint file should be the last item of
    # `all_model_checkpoint_paths`, according to the CheckpointState protobuf
    # definition.
    ckpt = tf.train.get_checkpoint_state(config.train.model_dir)
    if not ckpt or not ckpt.all_model_checkpoint_paths:
        raise ValueError('Could not find checkpoint in {}.'.format(
            config.train.model_dir
        ))

    # TODO: Any other way to get the global_step?
    checkpoints = [
        {'global_step': int(path.split('-')[-1]), 'file': path}
        for path in ckpt.all_model_checkpoint_paths
    ]

    # Get the run name from the checkpoint path. Do it before filtering the
    # list, as it may end up empty.
    # TODO: Can't it be set somewhere else?
    config.train.run_name = os.path.split(
        os.path.dirname(checkpoints[0]['file'])
    )[-1]

    if from_global_step is not None:
        checkpoints = [
            ckpt for ckpt in checkpoints
            if ckpt['global_step'] > from_global_step
        ]

        tf.logging.info(
            'Found %s checkpoints in model_dir with global_step > %s',
            len(checkpoints), from_global_step,
        )

    else:
        tf.logging.info(
            'Found {} checkpoints in model_dir'.format(len(checkpoints))
        )

    return checkpoints


def evaluate_once(config, saver, ops, checkpoint):
    """Run the evaluation once.

    # TODO: Also creates saver.
    Create a new session with the previously-built graph, run it through the
    dataset, calculate the evaluation metrics and write the corresponding
    summaries.

    Args:
        config: Config object for the model.
        saver: Saver object to restore checkpoint parameters.
        ops (dict): All the operations needed to successfully run the model.
            Expects the following keys: ``init_op``, ``metric_ops``,
            ``pred_objects``, ``pred_objects_classes``,
            ``pred_objects_scores``, ``train_filename``, ``train_objects``,
            ``losses`.
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
        'filenames': [],  # Filenames. TODO: Remove.
    }

    # TODO: Get runname from model-dir
    summary_dir = os.path.join(config.train.log_dir, config.train.run_name)

    with tf.Session() as sess:
        sess.run(ops['init_op'])
        saver.restore(sess, checkpoint['file'])

        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                (
                    _, batch_bboxes, batch_classes, batch_scores,
                    batch_filenames, batch_gt_objects,
                ) = sess.run([
                    ops['metric_ops'], ops['pred_objects'],
                    ops['pred_objects_classes'], ops['pred_objects_scores'],
                    ops['train_filename'], ops['train_objects'],
                ])

                output_per_batch['bboxes'].append(batch_bboxes)
                output_per_batch['classes'].append(batch_classes)
                output_per_batch['scores'].append(batch_scores)

                output_per_batch['gt_bboxes'].append(batch_gt_objects[:, :4])
                output_per_batch['gt_classes'].append(batch_gt_objects[:, 4])

                output_per_batch['filenames'].append(batch_filenames)

                val_losses = sess.run(ops['losses'])

        except tf.errors.OutOfRangeError:

            # Save final evaluation stats into summary under the checkpoint's
            # global step.
            map_0_5, per_class_0_5 = calculate_map(
                output_per_batch, config.network.num_classes, 0.5
            )

            # TODO: Find a way to generate these summaries automatically, or
            # less manually.
            summary = [
                tf.Summary.Value(tag='metrics/mAP@0.5', simple_value=map_0_5),
            ]

            for loss_name, loss_value in val_losses.items():
                summary.append(tf.Summary.Value(
                    tag=loss_name,
                    simple_value=loss_value
                ))

            for idx, val in enumerate(per_class_0_5):
                summary.append(tf.Summary.Value(
                    tag='metrics/AP@0.5/{}'.format(idx),
                    simple_value=val
                ))

            writer.add_summary(
                tf.Summary(value=summary),
                checkpoint['global_step']
            )

        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)


def calculate_map(output_per_batch, num_classes, iou_threshold=0.5):
    """Calculates mAP@iou_threshold from the detector's output.

    The procedure for calculating the average precision for class ``C`` is as
    follows (see `VOC mAP metric`_ for more details):

    Start by ranking all the predictions (for a given image and said class) in
    order of confidence.  Each of these predictions is marked as correct (true
    positive, when it has a IoU-threshold greater or equal to `iou_threshold`)
    or incorrect (false positive, in the other case).  This matching is
    performed greedily over the confidence scores, so a higher-confidence
    prediction will be matched over another lower-confidence one even if the
    latter has better IoU.  Also, each prediction is matched at most once, so
    repeated detections are counted as false positives.

    We then integrate over the interpolated PR curve, thus obtaining the value
    for the class' average precision.  This interpolation makes sure the
    precision curve is monotonically decreasing; for this, at each recall point
    ``r``, the precision is the maximum precision value among all recalls
    higher than ``r``.  The integration is performed over 11 fixed points over
    the curve (``[0.0, 0.1, ..., 1.0]``).

    Average the result among all the classes to obtain the final, ``mAP``,
    value.

    Args:
        output_per_batch (dict): Output of the detector to calculate mAP.
            Expects the following keys: ``bboxes``, ``classes``, ``scores``,
            ``gt_bboxes``, ``gt_classes``, ``filenames``.  Under each key,
            there should be a list of the results per batch as returned by the
            detector.
        num_classes (int): Number of classes on the dataset.
        threshold (float): IoU threshold for considering a match.

    Returns:
        (``np.float``, ``ndarray``) tuple. The first value is the mAP, while
        the second is an array of size (`num_classes`,), with the AP value per
        class.

    Note:
        The "difficult example" flag of VOC dataset is being ignored.

    Todo:
        * Use VOC2012-style for integrating the curve. That is, use all recall
          points instead of a fixed number of points like in VOC2007.

    .. _VOC mAP metric:
        http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf
    """
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
            is_detected = np.zeros(num_gt)

            # TP/FP labels for detected bboxes of (class, image).
            tp_fp_labels = np.zeros(len(sorted_indices))

            if num_gt == 0:
                # If no ground truth examples for class, all predictions must
                # be false positives.
                tp_fp_labels_by_class[cls].append(
                    (tp_fp_labels, cls_scores[sorted_indices])
                )
                continue

            # Get the IoUs for the class' bboxes.
            ious = bbox_overlaps(cls_bboxes, cls_gt_bboxes)

            # Greedily assign bboxes to ground truths (highest score first).
            for bbox_idx in sorted_indices:
                gt_match = np.argmax(ious[bbox_idx, :])
                if ious[bbox_idx, gt_match] >= iou_threshold:
                    # Over IoU threshold.
                    if not is_detected[gt_match]:
                        # And first detection: it's a true positive.
                        tp_fp_labels[bbox_idx] = True
                        is_detected[gt_match] = True

            tp_fp_labels_by_class[cls].append(
                (tp_fp_labels, cls_scores[sorted_indices])
            )

    # Calculate average precision per class.
    ap_per_class = np.zeros(num_classes)
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
        true_positives = labels[sorted_indices]
        false_positives = 1 - true_positives

        cum_true_positives = np.cumsum(true_positives)
        cum_false_positives = np.cumsum(false_positives)

        recall = cum_true_positives.astype(float) / num_examples
        precision = np.divide(
            cum_true_positives.astype(float),
            cum_true_positives + cum_false_positives
        )

        # Find AP by integrating over PR curve, with interpolated precision.
        ap = 0
        for t in np.linspace(0, 1, 11):
            if not np.any(recall >= t):
                # Recall is never higher than `t`, continue.
                continue
            ap += np.max(precision[recall >= t]) / 11  # Interpolated.

        ap_per_class[cls] = ap

    # Finally, mAP.
    mean_ap = np.mean(ap_per_class)

    return mean_ap, ap_per_class
