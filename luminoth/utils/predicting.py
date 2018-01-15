import json
import numpy as np
import os
import tensorflow as tf
import time

from PIL import Image

from luminoth.models import get_model
from luminoth.datasets import get_dataset
from luminoth.utils.config import get_config


def get_predictions(image_paths, config_files):
    """
    Get predictions for multiple images.

    When predicting many images we don't want to load the checkpoint each time.
    We load the checkpoint in the first iteration and then use the same
    session and graph for subsequent images.
    """
    config = get_config(config_files)

    if config.dataset.dir:
        # Gets the names of the classes
        classes_file = os.path.join(config.dataset.dir, 'classes.json')
        if tf.gfile.Exists(classes_file):
            class_labels = json.load(tf.gfile.GFile(classes_file))
        else:
            class_labels = None

    session = None
    fetches = None
    image_tensor = None

    for image_path in image_paths:
        with tf.gfile.Open(image_path, 'rb') as im_file:
            try:
                image = Image.open(im_file).convert('RGB')
            except tf.errors.OutOfRangeError as e:
                yield {
                    'error': '{}'.format(e),
                    'image_path': image_path,
                }
                continue

        preds = get_prediction(
            image, config,
            session=session, fetches=fetches,
            image_tensor=image_tensor, class_labels=class_labels,
            return_tf_vars=True
        )

        if session is None:
            # After first loop
            session = preds['session']
            fetches = preds['fetches']
            image_tensor = preds['image_tensor']

        yield {
            'objects': preds['objects'],
            'objects_labels': preds['objects_labels'],
            'objects_labels_prob': preds['objects_labels_prob'],
            'inference_time': preds['inference_time'],
            'image_path': image_path,
        }


def get_prediction(image, config, total=None, session=None,
                   fetches=None, image_tensor=None, class_labels=None,
                   return_tf_vars=False):
    """
    Gets the prediction given by the model `model_type` of the image `image`.
    If a checkpoint exists in the job's directory, load it.
    The names of the classes will be obtained from the dataset directory.
    Returns a dictionary with the objects, their labels and probabilities,
    the inference time and the scale factor. Also if the `return_tf_vars` is
    True, returns the image tensor, the entire prediction of the model and
    the sesssion.
    """

    if session is None and fetches is None and image_tensor is None:
        # Don't use data augmentation in predictions
        config.dataset.data_augmentation = None

        dataset_class = get_dataset(config.dataset.type)
        model_class = get_model(config.model.type)
        dataset = dataset_class(config)
        model = model_class(config)

        graph = tf.Graph()
        session = tf.Session(graph=graph)

        with graph.as_default():
            image_tensor = tf.placeholder(tf.float32, (None, None, 3))
            image_tf, _, process_meta = dataset.preprocess(image_tensor)
            pred_dict = model(image_tf)

            # Restore checkpoint
            if config.train.job_dir:
                job_dir = config.train.job_dir
                if config.train.run_name:
                    job_dir = os.path.join(job_dir, config.train.run_name)
                ckpt = tf.train.get_checkpoint_state(job_dir)
                if not ckpt or not ckpt.all_model_checkpoint_paths:
                    raise ValueError('Could not find checkpoint in {}.'.format(
                        job_dir
                    ))
                ckpt = ckpt.all_model_checkpoint_paths[-1]
                saver = tf.train.Saver(sharded=True, allow_empty=True)
                saver.restore(session, ckpt)
                tf.logging.info('Loaded checkpoint.')
            else:
                # A prediction without checkpoint is just used for testing
                tf.logging.warning(
                    'Could not load checkpoint. Using initialized model.')
                init_op = tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                )
                session.run(init_op)

            if config.model.network.with_rcnn:
                cls_prediction = pred_dict['classification_prediction']
                objects_tf = cls_prediction['objects']
                objects_labels_tf = cls_prediction['labels']
                objects_labels_prob_tf = cls_prediction['probs']
            else:
                rpn_prediction = pred_dict['rpn_prediction']
                objects_tf = rpn_prediction['proposals']
                objects_labels_prob_tf = rpn_prediction['scores']
                # All labels without RCNN are zero
                objects_labels_tf = tf.zeros(
                    tf.shape(objects_labels_prob_tf), dtype=tf.int32
                )

            fetches = {
                'objects': objects_tf,
                'labels': objects_labels_tf,
                'probs': objects_labels_prob_tf,
                'scale_factor': process_meta['scale_factor']
            }

            # If in debug mode, return the full prediction dictionary.
            if config.train.debug:
                fetches['_debug'] = pred_dict

    elif session is None or fetches is None or image_tensor is None:
        raise ValueError(
            'Either all `session`, `fetches` and `image_tensor` are None, '
            'or neither of them are.'
        )

    start_time = time.time()
    fetched = session.run(fetches, feed_dict={
        image_tensor: np.array(image)
    })
    end_time = time.time()

    objects = fetched['objects']
    objects_labels = fetched['labels']
    objects_labels_prob = fetched['probs']
    scale_factor = fetched['scale_factor']

    objects_labels = objects_labels.tolist()

    if class_labels is not None:
        objects_labels = [class_labels[obj] for obj in objects_labels]

    # Scale objects to original image dimensions
    objects /= scale_factor

    objects = objects.tolist()
    objects_labels_prob = objects_labels_prob.tolist()

    if total is not None:
        objects = objects[:total]
        objects_labels = objects_labels[:total]
        objects_labels_prob = objects_labels_prob[:total]

    res = {
        'objects': objects,
        'objects_labels': objects_labels,
        'objects_labels_prob': objects_labels_prob,
        'inference_time': end_time - start_time,
    }

    if return_tf_vars:
        res['image_tensor'] = image_tensor
        res['fetches'] = fetches
        res['session'] = session

    return res
