import json
import numpy as np
import os
import tensorflow as tf
import time

from PIL import Image

from luminoth.models import get_model
from luminoth.utils.config import get_model_config, load_config


def resize_image(image, min_size, max_size):
    """
    Resizes `image` if it's necesary
    """
    min_dimension = min(image.height, image.width)
    upscale = max(min_size / min_dimension, 1.)

    max_dimension = max(image.height, image.width)
    downscale = min(max_size / max_dimension, 1.)

    new_width = int(upscale * downscale * image.width)
    new_height = int(upscale * downscale * image.height)

    image = image.resize((new_width, new_height))
    image_array = np.array(image)[:, :, :3]  # TODO Read RGB
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, upscale * downscale


def get_predictions(image_paths, config_files):
    """
    Get predictions for multiple images.

    When predicting many images we don't want to load the checkpoint each time.
    We load the checkpoint in the first iteration and then use the same
    session and graph for subsequent images.
    """
    session = None
    fetches = None
    image_tensor = None

    predictions = []

    for image_path in image_paths:
        with tf.gfile.Open(image_path, 'rb') as im_file:
            try:
                image = Image.open(im_file)
            except tf.errors.OutOfRangeError as e:
                predictions.append({
                    'error': '{}'.format(e),
                    'image_path': image_path,
                })
                continue

        preds = get_prediction(
            image, config_files, return_tf_vars=True,
            session=session, fetches=fetches,
            image_tensor=image_tensor
        )

        if session is None:
            # After first loop
            session = preds['session']
            fetches = preds['fetches']
            image_tensor = preds['image_tensor']

        predictions.append({
            'objects': preds['objects'],
            'objects_labels': preds['objects_labels'],
            'objects_labels_prob': preds['objects_labels_prob'],
            'inference_time': preds['inference_time'],
            'scale_factor': preds['scale_factor'],
            'image_path': image_path,
        })

    return predictions


def get_prediction(image, config_files, session=None,
                   fetches=None, image_tensor=None,
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
    custom_config = load_config(config_files)
    model_class = get_model(custom_config.model.type)
    config = get_model_config(
        model_class.base_config, custom_config, None
    )

    if session is None and fetches is None and image_tensor is None:
        graph = tf.Graph()
        session = tf.Session(graph=graph)

        with graph.as_default():
            image_tensor = tf.placeholder(tf.float32, (1, None, None, 3))
            model = model_class(model_class.base_config)
            pred_dict = model(image_tensor)

            # Restore checkpoint
            if config.train.job_dir:
                ckpt = tf.train.get_checkpoint_state(config.train.job_dir)
                if not ckpt or not ckpt.all_model_checkpoint_paths:
                    raise ValueError('Could not find checkpoint in {}.'.format(
                        config.train.job_dir
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
            }
    elif session is None or fetches is None or image_tensor is None:
        raise ValueError(
            'Either all `session`, `fetches` and `image_tensor` are None, '
            'or neither of them are.'
        )

    image_resize_config = model_class.base_config.dataset.image_preprocessing

    image_array, scale_factor = resize_image(
        image, float(image_resize_config.min_size),
        float(image_resize_config.max_size)
    )

    start_time = time.time()
    fetched = session.run(fetches, feed_dict={
        image_tensor: image_array
    })
    end_time = time.time()

    objects = fetched['objects']
    objects_labels = fetched['labels']
    objects_labels_prob = fetched['probs']

    objects_labels = objects_labels.tolist()

    if config.dataset.dir:
        # Gets the names of the classes
        classes_file = os.path.join(config.dataset.dir, 'classes.json')
        if tf.gfile.Exists(classes_file):
            class_labels = json.load(tf.gfile.GFile(classes_file))
            objects_labels = [class_labels[obj] for obj in objects_labels]

    res = {
        'objects': objects.tolist(),
        'objects_labels': objects_labels,
        'objects_labels_prob': objects_labels_prob.tolist(),
        'inference_time': end_time - start_time,
        'scale_factor': scale_factor,
    }

    if return_tf_vars:
        res['image_tensor'] = image_tensor
        res['fetches'] = fetches
        res['session'] = session

    return res
