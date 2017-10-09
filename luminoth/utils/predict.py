import easydict
import json
import os
import time
import yaml

import numpy as np
import tensorflow as tf

from luminoth.models import get_model


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


def get_prediction(model_name, image, config_file, session=None, output=None,
                   image_tensor=None):
    """
    Gets the prediction given by the model `model_name` of the image `image`.
    TODO
    If `checkpoint_file` is not None, loads it and also loads the name of the
    classes if a `classes_file` is not None, and returns them.
    """
    model_class = get_model(model_name)
    config = easydict.EasyDict(yaml.load(tf.gfile.GFile(config_file)))
    if session and output and image_tensor:
        pass
    else:
        graph = tf.Graph()
        session = tf.Session(graph=graph)

        with graph.as_default():
            image_tensor = tf.placeholder(tf.float32, (1, None, None, 3))
            model = model_class(model_class.base_config)
            output = model(image_tensor)
            if config.train.job_dir and config.train.run_name:
                checkpoint_dir = os.path.join(
                    '.',
                    config.train.job_dir, config.train.run_name,
                    'model.ckpt-862094')
                saver = tf.train.Saver(sharded=True, allow_empty=True)
                saver.restore(session, checkpoint_dir)
            else:
                print('else')
                init_op = tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                )
                session.run(init_op)

    classification_prediction = output['classification_prediction']
    objects_tf = classification_prediction['objects']
    objects_labels_tf = classification_prediction['labels']
    objects_labels_prob_tf = classification_prediction['probs']
    image_resize_config = model_class.base_config.dataset.image_preprocessing

    image_array, scale_factor = resize_image(
        image, float(image_resize_config.min_size),
        float(image_resize_config.max_size)
    )

    start_time = time.time()
    objects, objects_labels, objects_labels_prob = session.run([
        objects_tf, objects_labels_tf, objects_labels_prob_tf
    ], feed_dict={
        image_tensor: image_array
    })
    end_time = time.time()

    if config.dataset.dir:
        classes_file = os.path.join(config.dataset.dir, 'classes.json')
        # Gets the names of the classes
        class_labels = json.load(tf.gfile.GFile(classes_file))
        objects_labels = [class_labels[obj] for obj in objects_labels]

    else:
        objects_labels = objects_labels.tolist()

    return {
        'objects': objects.tolist(),
        'objects_labels': objects_labels,
        'objects_labels_prob': objects_labels_prob.tolist(),
        'inference_time': end_time - start_time,
        'scale_factor': scale_factor,
        'image_tensor': image_tensor,
        'output': output,
        'session': session
    }
