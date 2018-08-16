"""Module where all generic task models are located.

Each class corresponds to a different task, providing a task-specific API for
ease of use. This API is common among different implementations, abstracting
away the pecularities of each task model. Thus, no knowledge of the inner
workings of said models should be needed to use any of these classes.
"""
from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.predicting import PredictorNetwork


class Detector(object):
    """Encapsulates an object detection model behavior.

    In order to perform object detection with a model implemented within
    Luminoth, this class should be used.

    Attributes:
        classes (list of str): Ordered class names for the detector.
        prob (float): Default probability threshold for predictions.

    TODO:
        - Don't create a TF session internally (or make its creation optional)
          in order to be compatible with both TF-Eager and Jupyter Notebooks.
        - Manage multiple instantiations correctly in order to avoid creating
          the same TF objects over and over (which appends the `_N` suffix to
          the graph and makes the checkpoint loading fail).
    """

    DEFAULT_CHECKPOINT = 'accurate'

    def __init__(self, checkpoint=None, config=None, prob=0.7, classes=None):
        """Instantiate a detector object with the appropriate config.

        Arguments:
            checkpoint (str): Checkpoint id or alias to instantiate the
                detector as.
            config (dict): Configuration parameters describing the desired
                model. See `get_config` to load a config file.

        Note:
            Only one of the parameters must be specified. If none is, we
            default to loading the checkpoint indicated by
            `DEFAULT_CHECKPOINT`.
        """
        if checkpoint is not None and config is not None:
            raise ValueError(
                'Only one of `checkpoint` or `config` must be specified in '
                'order to instantiate a Detector.'
            )

        if checkpoint is None and config is None:
            # Neither checkpoint no config specified, default to
            # `DEFAULT_CHECKPOINT`.
            checkpoint = self.DEFAULT_CHECKPOINT

        if checkpoint:
            config = get_checkpoint_config(checkpoint)

        # Prevent the model itself from filtering its proposals (default
        # value of 0.5 is in use in the configs).
        # TODO: A model should always return all of its predictions. The
        # filtering should be done (if at all) by PredictorNetwork.
        if config.model.type == 'fasterrcnn':
            config.model.rcnn.proposals.min_prob_threshold = 0.0
        elif config.model.type == 'ssd':
            config.model.proposals.min_prob_threshold = 0.0

        # TODO: Remove dependency on `PredictorNetwork` or clearly separate
        # responsibilities.
        self._network = PredictorNetwork(config)

        self.prob = prob

        # Use the labels when available, integers when not.
        self._model_classes = (
            self._network.class_labels if self._network.class_labels
            else list(range(config.model.network.num_classes))
        )
        if classes:
            self.classes = set(classes)
            if not set(self._model_classes).issuperset(self.classes):
                raise ValueError(
                    '`classes` must be contained in the detector\'s classes. '
                    'Available classes are: {}.'.format(self._model_classes)
                )
        else:
            self.classes = set(self._model_classes)

    def predict(self, images, prob=None, classes=None):
        """Run the detector through a set of images.

        Arguments:
            images (numpy.ndarray or list): Either array of dimensions
                `(height, width, channels)` (single image) or array of
                dimensions `(number_of_images, height, width, channels)`
                (multiple images). If a list, must be a list of rank 3 arrays.
            prob (float): Override configured probability threshold for
                predictions.
            classes (set of str): Override configured class names to consider.

        Returns:
            Either list of objects detected in the image (single image case) or
            list of list of objects detected (multiple images case).

            In the multiple images case, the outer list has `number_of_images`
            elements, while the inner ones have the number of objects detected
            in each image.

            Each object has the format::

                {
                    'bbox': [x_min, y_min, x_max, y_max],
                    'label': '<cat|dog|person|...>' | 0..C,
                    'prob': prob
                }

            The coordinates are integers, where `(x_min, y_min)` are the
            coordinates of the top-left corner of the bounding box, while
            `(x_max, y_max)` the bottom-right. By convention, the top-left
            corner of the image is coordinate `(0, 0)`.

            The probability, `prob`, is a float between 0 and 1, indicating the
            confidence of the detection being correct.

            The label of the object, `label`, may be either a string if the
            classes file for the model is found or an integer between 0 and the
            number of classes `C`.

        """
        # If it's a single image (ndarray of rank 3), turn into a list.
        single_image = False
        if not isinstance(images, list):
            if len(images.shape) == 3:
                images = [images]
                single_image = True

        if prob is None:
            prob = self.prob

        if classes is None:
            classes = self.classes
        else:
            classes = set(classes)

        # TODO: Remove the loop once (and if) we implement batch sizes. Neither
        # Faster R-CNN nor SSD support batch size yet, so it's the same for
        # now.
        predictions = []
        for image in images:
            predictions.append([
                pred for pred in self._network.predict_image(image)
                if pred['prob'] >= prob and pred['label'] in classes
            ])

        if single_image:
            predictions = predictions[0]

        return predictions
