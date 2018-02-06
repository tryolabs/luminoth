import numpy as np
import sonnet as snt
import tensorflow as tf

from luminoth.models.fasterrcnn.rcnn import RCNN
from luminoth.models.fasterrcnn.rpn import RPN
from luminoth.models.base import TruncatedBaseNetwork
from luminoth.utils.anchors import generate_anchors_reference
from luminoth.utils.vars import VAR_LOG_LEVELS, variable_summaries, get_saver


class FasterRCNN(snt.AbstractModule):
    """Faster RCNN Network module

    Builds the Faster RCNN network architecture using different submodules.
    Calculates the total loss of the model based on the different losses by
    each of the submodules.

    It is also responsible for building the anchor reference which is used in
    graph for generating the dynamic anchors.
    """
    def __init__(self, config, name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)

        # Main configuration object, it holds not only the necessary
        # information for this module but also configuration for each of the
        # different submodules.
        self._config = config

        # Total number of classes to classify. If not using RCNN then it is not
        # used. TODO: Make it *more* optional.
        self._num_classes = config.model.network.num_classes

        # Generate network with RCNN thus allowing for classification of
        # objects and not just finding them.
        self._with_rcnn = config.model.network.with_rcnn

        # Turn on debug mode with returns more Tensors which can be used for
        # better visualization and (of course) debugging.
        self._debug = config.train.debug
        self._seed = config.train.seed

        # Anchor config, check out the docs of base_config.yml for a better
        # understanding of how anchors work.
        self._anchor_base_size = config.model.anchors.base_size
        self._anchor_scales = np.array(config.model.anchors.scales)
        self._anchor_ratios = np.array(config.model.anchors.ratios)
        self._anchor_stride = config.model.anchors.stride

        # Anchor reference for building dynamic anchors for each image in the
        # computation graph.
        self._anchor_reference = generate_anchors_reference(
            self._anchor_base_size, self._anchor_ratios, self._anchor_scales
        )

        # Total number of anchors per point.
        self._num_anchors = self._anchor_reference.shape[0]

        # Weights used to sum each of the losses of the submodules
        self._rpn_cls_loss_weight = config.model.loss.rpn_cls_loss_weight
        self._rpn_reg_loss_weight = config.model.loss.rpn_reg_loss_weights

        self._rcnn_cls_loss_weight = config.model.loss.rcnn_cls_loss_weight
        self._rcnn_reg_loss_weight = config.model.loss.rcnn_reg_loss_weights
        self._losses_collections = ['fastercnn_losses']

        # We want the pretrained model to be outside the FasterRCNN name scope.
        self.base_network = TruncatedBaseNetwork(config.model.base_network)

    def _build(self, image, gt_boxes=None, is_training=False):
        """
        Returns bounding boxes and classification probabilities.

        Args:
            image: A tensor with the image.
                Its shape should be `(height, width, 3)`.
            gt_boxes: A tensor with all the ground truth boxes of that image.
                Its shape should be `(num_gt_boxes, 5)`
                Where for each gt box we have (x1, y1, x2, y2, label),
                in that order.
            is_training: A boolean to whether or not it is used for training.

        Returns:
            classification_prob: A tensor with the softmax probability for
                each of the bounding boxes found in the image.
                Its shape should be: (num_bboxes, num_categories + 1)
            classification_bbox: A tensor with the bounding boxes found.
                It's shape should be: (num_bboxes, 4). For each of the bboxes
                we have (x1, y1, x2, y2)
        """
        if gt_boxes is not None:
            gt_boxes = tf.cast(gt_boxes, tf.float32)
        # A Tensor with the feature map for the image,
        # its shape should be `(feature_height, feature_width, 512)`.
        # The shape depends of the pretrained network in use.

        # Set rank and last dimension before using base network
        # TODO: Why does it loose information when using queue?
        image.set_shape((None, None, 3))

        conv_feature_map = self.base_network(
            tf.expand_dims(image, 0), is_training=is_training
        )

        # The RPN submodule which generates proposals of objects.
        self._rpn = RPN(
            self._num_anchors, self._config.model.rpn,
            debug=self._debug, seed=self._seed
        )
        if self._with_rcnn:
            # The RCNN submodule which classifies RPN's proposals and
            # classifies them as background or a specific class.
            self._rcnn = RCNN(
                self._num_classes, self._config.model.rcnn,
                debug=self._debug, seed=self._seed
            )

        image_shape = tf.shape(image)[0:2]

        variable_summaries(
            conv_feature_map, 'conv_feature_map', 'reduced'
        )

        # Generate anchors for the image based on the anchor reference.
        all_anchors = self._generate_anchors(tf.shape(conv_feature_map))
        rpn_prediction = self._rpn(
            conv_feature_map, image_shape, all_anchors,
            gt_boxes=gt_boxes, is_training=is_training
        )

        prediction_dict = {
            'rpn_prediction': rpn_prediction,
        }

        if self._debug:
            prediction_dict['image'] = image
            prediction_dict['image_shape'] = image_shape
            prediction_dict['all_anchors'] = all_anchors
            prediction_dict['anchor_reference'] = tf.convert_to_tensor(
                self._anchor_reference
            )
            if gt_boxes is not None:
                prediction_dict['gt_boxes'] = gt_boxes
            prediction_dict['conv_feature_map'] = conv_feature_map

        if self._with_rcnn:
            proposals = tf.stop_gradient(rpn_prediction['proposals'])
            classification_pred = self._rcnn(
                conv_feature_map, proposals,
                image_shape, self.base_network,
                gt_boxes=gt_boxes, is_training=is_training
            )

            prediction_dict['classification_prediction'] = classification_pred

        return prediction_dict

    def loss(self, prediction_dict, return_all=False):
        """Compute the joint training loss for Faster RCNN.

        Args:
            prediction_dict: The output dictionary of the _build method from
                which we use two different main keys:

                rpn_prediction: A dictionary with the output Tensors from the
                    RPN.
                classification_prediction: A dictionary with the output Tensors
                    from the RCNN.

        Returns:
            If `return_all` is False, a tensor for the total loss. If True, a
            dict with all the internal losses (RPN's, RCNN's, regularization
            and total loss).
        """

        with tf.name_scope('losses'):
            rpn_loss_dict = self._rpn.loss(
                prediction_dict['rpn_prediction']
            )

            # Losses have a weight assigned, we multiply by them before saving
            # them.
            rpn_loss_dict['rpn_cls_loss'] = (
                rpn_loss_dict['rpn_cls_loss'] * self._rpn_cls_loss_weight)
            rpn_loss_dict['rpn_reg_loss'] = (
                rpn_loss_dict['rpn_reg_loss'] * self._rpn_reg_loss_weight)

            prediction_dict['rpn_loss_dict'] = rpn_loss_dict

            if self._with_rcnn:
                rcnn_loss_dict = self._rcnn.loss(
                    prediction_dict['classification_prediction']
                )

                rcnn_loss_dict['rcnn_cls_loss'] = (
                    rcnn_loss_dict['rcnn_cls_loss'] *
                    self._rcnn_cls_loss_weight
                )
                rcnn_loss_dict['rcnn_reg_loss'] = (
                    rcnn_loss_dict['rcnn_reg_loss'] *
                    self._rcnn_reg_loss_weight
                )

                prediction_dict['rcnn_loss_dict'] = rcnn_loss_dict
            else:
                rcnn_loss_dict = {}

            all_losses_items = (
                list(rpn_loss_dict.items()) + list(rcnn_loss_dict.items()))

            for loss_name, loss_tensor in all_losses_items:
                tf.summary.scalar(
                    loss_name, loss_tensor,
                    collections=self._losses_collections
                )
                # We add losses to the losses collection instead of manually
                # summing them just in case somebody wants to use it in another
                # place.
                tf.losses.add_loss(loss_tensor)

            # Regularization loss is automatically saved by TensorFlow, we log
            # it differently so we can visualize it independently.
            regularization_loss = tf.losses.get_regularization_loss()
            # Total loss without regularization
            no_reg_loss = tf.losses.get_total_loss(
                add_regularization_losses=False
            )
            total_loss = tf.losses.get_total_loss()

            tf.summary.scalar(
                'total_loss', total_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'no_reg_loss', no_reg_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'regularization_loss', regularization_loss,
                collections=self._losses_collections
            )

            if return_all:
                loss_dict = {
                    'total_loss': total_loss,
                    'no_reg_loss': no_reg_loss,
                    'regularization_loss': regularization_loss,
                }

                for loss_name, loss_tensor in all_losses_items:
                    loss_dict[loss_name] = loss_tensor

                return loss_dict

            # We return the total loss, which includes:
            # - rpn loss
            # - rcnn loss (if activated)
            # - regularization loss
            return total_loss

    def _generate_anchors(self, feature_map_shape):
        """Generate anchor for an image.

        Using the feature map, the output of the pretrained network for an
        image, and the anchor_reference generated using the anchor config
        values. We generate a list of anchors.

        Anchors are just fixed bounding boxes of different ratios and sizes
        that are uniformly generated throught the image.

        Args:
            feature_map_shape: Shape of the convolutional feature map used as
                input for the RPN. Should be (batch, height, width, depth).

        Returns:
            all_anchors: A flattened Tensor with all the anchors of shape
                `(num_anchors_per_points * feature_width * feature_height, 4)`
                using the (x1, y1, x2, y2) convention.
        """
        with tf.variable_scope('generate_anchors'):
            grid_width = feature_map_shape[2]  # width
            grid_height = feature_map_shape[1]  # height
            shift_x = tf.range(grid_width) * self._anchor_stride
            shift_y = tf.range(grid_height) * self._anchor_stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

            shift_x = tf.reshape(shift_x, [-1])
            shift_y = tf.reshape(shift_y, [-1])

            shifts = tf.stack(
                [shift_x, shift_y, shift_x, shift_y],
                axis=0
            )

            shifts = tf.transpose(shifts)
            # Shifts now is a (H x W, 4) Tensor

            # Expand dims to use broadcasting sum.
            all_anchors = (
                np.expand_dims(self._anchor_reference, axis=0) +
                tf.expand_dims(shifts, axis=1)
            )

            # Flatten
            all_anchors = tf.reshape(
                all_anchors, (-1, 4)
            )
            return all_anchors

    @property
    def summary(self):
        """
        Generate merged summary of all the sub-summaries used inside the
        Faster R-CNN network.
        """
        summaries = [
            tf.summary.merge_all(key='rpn'),
        ]

        summaries.append(
            tf.summary.merge_all(key=self._losses_collections[0])
        )

        if self._with_rcnn:
            summaries.append(tf.summary.merge_all(key='rcnn'))

        return tf.summary.merge(summaries)

    @property
    def vars_summary(self):
        return {
            key: tf.summary.merge_all(key=collection)
            for key, collections in VAR_LOG_LEVELS.items()
            for collection in collections
        }

    def get_trainable_vars(self):
        """Get trainable vars included in the module.
        """
        trainable_vars = snt.get_variables_in_module(self)
        if self._config.model.base_network.trainable:
            pretrained_trainable_vars = self.base_network.get_trainable_vars()
            if len(pretrained_trainable_vars):
                tf.logging.info(
                    'Training {} vars from pretrained module; '
                    'from "{}" to "{}".'.format(
                        len(pretrained_trainable_vars),
                        pretrained_trainable_vars[0].name,
                        pretrained_trainable_vars[-1].name,
                    )
                )
            else:
                tf.logging.info('No vars from pretrained module to train.')
            trainable_vars += pretrained_trainable_vars
        else:
            tf.logging.info('Not training variables from pretrained module')

        return trainable_vars

    def get_saver(self, ignore_scope=None):
        """Get an instance of tf.train.Saver for all modules and submodules.
        """
        return get_saver((self, self.base_network), ignore_scope=ignore_scope)

    def load_pretrained_weights(self):
        """Get operation to load pretrained weights from file.
        """
        return self.base_network.load_weights()
