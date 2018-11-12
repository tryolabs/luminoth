import json
import numpy as np
import os
import tensorflow as tf

from luminoth.models import get_model
from luminoth.datasets import get_dataset


class PredictorNetwork(object):
    """Instantiates a network in order to get predictions from it.

    If a checkpoint exists in the job's directory, load it.  The names of the
    classes will be obtained from the dataset directory.

    Returns a list of objects detected, which is a dict of its coordinates,
    label and probability, ordered by probability.
    """

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

            if config.model.type == 'ssd':
                cls_prediction = pred_dict['classification_prediction']
                objects_tf = cls_prediction['objects']
                objects_labels_tf = cls_prediction['labels']
                objects_labels_prob_tf = cls_prediction['probs']
            elif config.model.type == 'fasterrcnn':
                if config.model.network.get('with_rcnn', False):
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
            else:
                raise ValueError(
                    "Model type '{}' not supported".format(config.model.type)
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

    def predict_image(self, image):
        fetched = self.session.run(self.fetches, feed_dict={
            self.image_placeholder: np.array(image)
        })

        objects = fetched['objects']
        labels = fetched['labels'].tolist()
        probs = fetched['probs'].tolist()
        scale_factor = fetched['scale_factor']

        if self.class_labels is not None:
            labels = [self.class_labels[label] for label in labels]

        # Scale objects to original image dimensions
        if isinstance(scale_factor, tuple):
            # If scale factor is a tuple, it means we need to scale height and
            # width by a different amount. In that case scale factor is:
            # (scale_factor_height, scale_factor_width)
            objects /= [scale_factor[1], scale_factor[0],
                        scale_factor[1], scale_factor[0]]
        else:
            # If scale factor is a scalar, height and width get scaled by the
            # same amount
            objects /= scale_factor

        # Cast to int to consistently return the same type in Python 2 and 3
        objects = [
            [int(round(coord)) for coord in obj]
            for obj in objects.tolist()
        ]

        predictions = sorted([
            {
                'bbox': obj,
                'label': label,
                'prob': round(prob, 4),
            } for obj, label, prob in zip(objects, labels, probs)
        ], key=lambda x: x['prob'], reverse=True)

        return predictions
