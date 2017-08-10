import click
import numpy as np
import os
import tensorflow as tf

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
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')
def evaluate(model_type, dataset_split, config_file, model_dir, log_dir,
             override_params):
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

    # Override default dataset split
    config.dataset.split = dataset_split

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

    pretrained_dict = pretrained(train_image, is_training=False)
    prediction_dict = model(
        train_image, pretrained_dict['net'], train_objects, is_training=False
    )

    pred_objects = prediction_dict['classification_prediction']['objects']
    pred_objects_classes = prediction_dict['classification_prediction']['objects_labels']
    pred_objects_scores = prediction_dict['classification_prediction']['objects_labels_prob']

    # metrics(pred_objects, pred_objects_classes, train_objects)

    batch_loss = model.loss(prediction_dict)
    total_loss, _ = tf.metrics.mean(
        batch_loss, name='loss',
        metrics_collections='metrics',
        updates_collections='metric_ops',
    )
    # tf.summary.scalar('loss', total_loss)

    metric_ops = tf.get_collection('metric_ops')
    metrics = tf.get_collection('metrics')

    # summarizer = tf.summary.merge([
    #     tf.summary.merge_all(),
    #     model.summary,
    # ])

    last_checkpoint = tf.train.get_checkpoint_state(model_dir)
    if not last_checkpoint or not last_checkpoint.model_checkpoint_path:
        raise ValueError('Could not find checkpoint in {}.'.format(model_dir))

    config.train.run_name = os.path.split(
        os.path.dirname(last_checkpoint.model_checkpoint_path))[-1]

    global_step = int(last_checkpoint.model_checkpoint_path.split('-')[-1])
    tf.logging.info('Evaluating global_step {}'.format(global_step))

    last_checkpoint_path = last_checkpoint.model_checkpoint_path
    tf.logging.info('Using checkpoint "{}"'.format(last_checkpoint_path))
    config.train.checkpoint_file = last_checkpoint_path

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    saver = get_saver((model, pretrained, ))

    # TODO: Get runname from model-dir
    summary_dir = os.path.join(config.train.log_dir, config.train.run_name)

    all_bboxes = []
    all_classes = []
    all_scores = []
    all_gt_bboxes = []
    all_gt_classes = []
    all_filenames = []
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, config.train.checkpoint_file)

        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # summary, _ = sess.run([summarizer, metric_ops])
                (
                    _, batch_bboxes, batch_classes, batch_scores,
                    batch_filenames, batch_gt_objects,
                ) = sess.run([
                    metric_ops, pred_objects, pred_objects_classes,
                    pred_objects_scores, train_filename, train_objects,
                ])

                all_gt_bboxes.append(batch_gt_objects[:, :4])
                all_gt_classes.append(batch_gt_objects[:, 4])
                all_bboxes.append(batch_bboxes)
                all_classes.append(batch_classes)
                all_scores.append(batch_scores)
                all_filenames.append(batch_filenames)

                val_loss = sess.run(total_loss)
                print('val_loss = {:.2f}'.format(val_loss))

        except tf.errors.OutOfRangeError:

            # TODO: Do we want *everything* on the summaries or just
            # val_loss/mAP?
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='val_loss', simple_value=val_loss),
            ])
            writer.add_summary(summary, global_step)

        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)

    # Calculate mAP@iou with the results from running on the dataset.
    iou_threshold = 0.5

    # For each image, order predictions by score and classify as TP or FP.
    # TODO: Ignore difficult examples.
    # TODO: Use authoritative source of examples count.

    # List; first by class, then by example. Object is tuple of ndarrays of
    # size (D_{c,i},), for tp/fp labels and for score, where D_{c,i} is the
    # number of detected boxes for class `c` on image `i`.
    # TODO: Get number of classes from here or somewhere else?
    tp_fp_labels_by_class = [[] for _ in range(config.network.num_classes)]
    num_examples_per_class = [0 for _ in range(config.network.num_classes)]

    for idx in range(len(all_bboxes)):

        classes = all_classes[idx]  # D_{i}, number of detected for ith image.
        bboxes = all_bboxes[idx]  # (D_{i}, 4).
        scores = all_scores[idx]
        gt_bboxes = all_gt_bboxes[idx]
        gt_classes = all_gt_classes[idx]

        # Analysis must be made per-class.
        for cls in range(config.network.num_classes):
            cls_bboxes = bboxes[classes == cls, :]
            cls_scores = scores[classes == cls]
            cls_gt_bboxes = gt_bboxes[gt_classes == cls, :]

            num_gt = cls_gt_bboxes.shape[0]
            num_examples_per_class[cls] += num_gt

            sorted_indices = np.argsort(-cls_scores)  # Sort by score, descending.
            is_detected = np.zeros(num_gt)  # Has been previously detected.
            tp_fp_labels = np.zeros(len(sorted_indices))  # Labels for bboxes of class, image.

            if num_gt == 0:
                # If no ground truth examples for class, all flase positives.
                tp_fp_labels_by_class[cls].append(
                    (tp_fp_labels, cls_scores[sorted_indices])
                )
                continue

            # Get the IOUs for the classes bboxes.
            ious = bbox_overlaps(cls_bboxes, cls_gt_bboxes)

            # Greedily assign bboxes to ground truths (highest score first).
            for bbox_idx in sorted_indices:
                gt_match = np.argmax(ious[bbox_idx, :])
                if ious[bbox_idx, gt_match] >= iou_threshold:
                    # TODO: Check `is_difficult` for image too.
                    if not is_detected[gt_match]:
                        # First detection, it's a true positive.
                        tp_fp_labels[bbox_idx] = True
                        is_detected[gt_match] = True

            tp_fp_labels_by_class[cls].append(
                (tp_fp_labels, cls_scores[sorted_indices])
            )

    # Calculate AP per class.
    ap_per_class = np.zeros(config.network.num_classes)
    for cls in range(config.network.num_classes):
        tp_fp_labels = tp_fp_labels_by_class[cls]
        num_examples = num_examples_per_class[cls]

        # Flatten and sort the tp/fp by score.
        labels, scores = zip(*tp_fp_labels)
        labels = np.concatenate(labels)
        scores = np.concatenate(scores)

        # Calculate precision and recall.
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

        # Find AP by "integrating" over PR curve, with interpolated precision.
        # TODO: This is VOC2007 AP; 2012 uses *all* recall points.
        # TODO: See https://github.com/tensorflow/models/blob/a6df5573/object_detection/utils/metrics.py#L117 for the other way.
        ap = 0
        for t in np.linspace(0, 1, 11):
            if not np.any(recall >= t):
                # Recall is never higher than `t`, continue.
                continue

            prec = np.max(precision[recall > t])  # Interpolated.
            ap += prec / 11

        ap_per_class[cls] = ap

    # Finally, mAP.
    mean_ap = np.mean(np.array(ap_per_class))

    from IPython import embed; embed(display_banner=False)
