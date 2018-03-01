import json
import numpy as np
import os
import tensorflow as tf
import time

from luminoth.models import get_model
from luminoth.datasets import get_dataset


class PredictorNetwork(object):
    """Instantiates a network in order to get predictions from it

    If a checkpoint exists in the job's directory, load it.
    The names of the classes will be obtained from the dataset directory.
    Returns a dictionary with the objects, their labels and probabilities,
    the inference time and the scale factor."""

    def __init__(self, config):

        if config.dataset.dir:
            # Gets the names of the classes
            classes_file = os.path.join(config.dataset.dir, 'classes.json')
            if tf.gfile.Exists(classes_file):
                self.class_labels = json.load(tf.gfile.GFile(classes_file))
            else:
                self.class_labels = None

        # Don't use data augmentation in predictions
        config.dataset.data_augmentation = None

        dataset_class = get_dataset(config.dataset.type)
        model_class = get_model(config.model.type)
        dataset = dataset_class(config)
        model = model_class(config)

        graph = tf.Graph()
        self.session = tf.Session(graph=graph)

        with graph.as_default():
            self.image_placeholder = tf.placeholder(
                tf.float32, (None, None, 3)
            )
            image_tf, _, process_meta = dataset.preprocess(
                self.image_placeholder
            )
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
                saver.restore(self.session, ckpt)
                tf.logging.info('Loaded checkpoint.')
            else:
                # A prediction without checkpoint is just used for testing
                tf.logging.warning(
                    'Could not load checkpoint. Using initialized model.')
                init_op = tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                )
                self.session.run(init_op)

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

            self.fetches = {
                'objects': objects_tf,
                'labels': objects_labels_tf,
                'probs': objects_labels_prob_tf,
                'scale_factor': process_meta['scale_factor']
            }

            # If in debug mode, return the full prediction dictionary.
            if config.train.debug:
                self.fetches['_debug'] = pred_dict

    def predict_image(self, image, total_predictions=None):
        start_time = time.time()
        fetched = self.session.run(self.fetches, feed_dict={
            self.image_placeholder: np.array(image)
        })
        end_time = time.time()

        objects = fetched['objects']
        objects_labels = fetched['labels']
        objects_labels_prob = fetched['probs']
        scale_factor = fetched['scale_factor']

        objects_labels = objects_labels.tolist()

        if self.class_labels is not None:
            objects_labels = [self.class_labels[obj] for obj in objects_labels]

        # Scale objects to original image dimensions
        objects /= scale_factor

        objects = objects.tolist()
        objects_labels_prob = objects_labels_prob.tolist()

        if total_predictions:
            objects = objects[:total_predictions]
            objects_labels = objects_labels[:total_predictions]
            objects_labels_prob = objects_labels_prob[:total_predictions]

        return {
            'objects': objects,
            'objects_labels': objects_labels,
            'objects_labels_prob': objects_labels_prob,
            'inference_time': end_time - start_time,
        }
