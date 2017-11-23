import tensorflow as tf
import sonnet as snt
import numpy as np

from luminoth.models.base.fpn import FPN
from luminoth.models.retina.class_subnet import ClassSubnet
from luminoth.models.retina.box_subnet import BoxSubnet
from luminoth.models.retina.retina_target import RetinaTarget
from luminoth.models.retina.retina_proposal import RetinaProposal

from luminoth.utils.anchors import generate_anchors_reference
from luminoth.utils.bbox_transform_tf import decode
from luminoth.utils.image import adjust_bboxes
from luminoth.utils.losses import smooth_l1_loss, focal_loss
from luminoth.utils.vars import get_saver


class Retina(snt.AbstractModule):
    """Retina Netowrk module.

    Builds Retina (a.k.a. RetinaNet), a network for object detection based
    around the focal loss. Also calculates the training loss.
    """
    def __init__(self, config, name='retina'):
        super(Retina, self).__init__(name=name)

        self._config = config
        self._num_classes = config.model.network.num_classes

        self._anchor_base_size = config.model.anchors.base_size
        self._anchor_ratios = np.array(config.model.anchors.ratios)
        self._anchor_scales = np.array(config.model.anchors.scales)

        self._anchor_reference = generate_anchors_reference(
            self._anchor_base_size, self._anchor_ratios, self._anchor_scales
        )
        # Total number of anchors per position in a level.
        self._num_anchors = self._anchor_reference.shape[0]

        self._debug = config.train.debug
        self._seed = config.train.seed

        self.fpn = FPN(
            config.model.fpn, parent_name=name
        )

        self._losses_collections = ['retina_losses']
        self._reduce_sum = self._config.model.loss.reduce_sum
        self._gamma = self._config.model.loss.gamma
        self._class_weight = float(self._config.model.loss.class_weight)
        self._background_divider = float(
            self._config.model.loss.background_weight_divider)
        self._use_softmax = self._config.model.class_subnet.final.use_softmax

        self._offset = self._config.model.anchors.offset

        self._share_weights = self._config.model.share_weights

    def _build(self, image, gt_boxes=None, is_training=True):
        """
        Args:
            image: (height, width, 3)
            gt_boxes: (num_gt_boxes, 4) or None.
            is_training: Bool.

        Returns:
            classification_prediction: Actual predictions.
                objects: (num_objects, 4)
                labels: (num_objects,)
                probs: (num_objects,)
            pre_nms_prediction: For computing loss and debugging.
                cls_scores: (num_preds,)
                bbox_preds: (num_preds, 4)
                cls_target: If training. (num_preds,)
                bbox_target: If training. (num_preds, 4)
        """
        im_shape_int = tf.shape(image, name='im_shape')[:2]
        im_shape = tf.to_float(im_shape_int, name='im_shape_float')
        fpn_levels = self.fpn(
            tf.expand_dims(image, axis=0), is_training=is_training
        )

        # Add new FPN levels (AMIP)
        fpn_levels = self._add_fpn_levels(fpn_levels)

        if self._share_weights:
            box_subnet = BoxSubnet(
                self._config.model.box_subnet, num_anchors=self._num_anchors
            )
            class_subnet = ClassSubnet(
                self._config.model.class_subnet, num_anchors=self._num_anchors,
                num_classes=self._num_classes
            )
        self._proposal = RetinaProposal(
            self._config.model.proposal, num_classes=self._num_classes
        )

        bbox_preds = []
        class_scores = []
        all_anchors = []
        for level in fpn_levels:
            if not self._share_weights:
                box_subnet = BoxSubnet(
                    self._config.model.box_subnet,
                    num_anchors=self._num_anchors
                )
                class_subnet = ClassSubnet(
                    self._config.model.class_subnet,
                    num_anchors=self._num_anchors,
                    num_classes=self._num_classes
                )
            level_shape_int = tf.shape(level, name='level_shape')[1:3]
            level_shape = tf.to_float(level_shape_int, name='level_shape_int')
            anchors = self._generate_anchors(level_shape, im_shape)
            level_bbox_pred_bank = box_subnet(level)
            level_bbox_preds = tf.reshape(
                level_bbox_pred_bank,
                shape=[-1, 4]
            )
            level_class_score_bank = class_subnet(level)
            level_class_scores = tf.reshape(
                level_class_score_bank,
                shape=[-1, self._num_classes + 1]
            )

            # Get rid of proposals from anchors outside the image.
            xmin, ymin, xmax, ymax = tf.split(anchors, 4, axis=1)
            proposals_filter = tf.logical_and(
                tf.logical_and(
                    tf.greater_equal(xmin, 0 - self._offset),
                    tf.greater_equal(ymin, 0 - self._offset),
                ),
                tf.logical_and(
                    tf.less_equal(xmax, im_shape_int[1] + self._offset),
                    tf.less_equal(ymax, im_shape_int[0] + self._offset),
                ),
            )
            proposals_filter = tf.reshape(proposals_filter, [-1])
            level_bbox_preds = tf.boolean_mask(
                level_bbox_preds, proposals_filter, name='mask_bbox_preds'
            )
            level_class_scores = tf.boolean_mask(
                level_class_scores, proposals_filter, name='mask_scores'
            )
            anchors = tf.boolean_mask(
                anchors, proposals_filter, name='mask_anchors'
            )

            bbox_preds.append(level_bbox_preds)
            class_scores.append(level_class_scores)
            all_anchors.append(anchors)

        class_scores = tf.concat(class_scores, axis=0)

        bbox_preds = tf.concat(bbox_preds, axis=0)
        all_anchors = tf.concat(all_anchors, axis=0)

        pred_dict = {
            'pre_nms_prediction': {
                'cls_scores': class_scores,
                'bbox_preds': bbox_preds,
            }
        }

        proposals = decode(all_anchors, bbox_preds)
        proposal_dict = self._proposal(
            class_scores, proposals, all_anchors
        )

        pred_dict['classification_prediction'] = {
            'objects': proposal_dict['objects'],
            'labels': proposal_dict['proposal_label'],
            'probs': proposal_dict['proposal_label_prob'],
        }

        if gt_boxes is not None:
            gt_boxes = tf.to_float(gt_boxes)
            retina_target = RetinaTarget(
                self._config.model.target, num_classes=self._num_classes,
            )
            cls_target, bbox_target = retina_target(
                all_anchors, gt_boxes
            )
            pred_dict['pre_nms_prediction']['cls_target'] = cls_target
            pred_dict['pre_nms_prediction']['bbox_target'] = bbox_target
        return pred_dict

    def loss(self, pred_dict, return_all=False):
        """Compute training loss for object detection with Retina.

        We use focal loss for classification, and smooth L1 loss for bbox
        regression.

        Args:
            pred_dict: Output of _build()
            return_all: For compliance with the luminoth API.
                Ignored internally.

        Returns:
            total_loss
        """
        with tf.name_scope('losses'):
            pred_dict = pred_dict['pre_nms_prediction']
            cls_scores = pred_dict['cls_scores']
            cls_target = pred_dict['cls_target']

            bbox_preds = pred_dict['bbox_preds']
            bbox_target = pred_dict['bbox_target']

            # Don't consider boxes with target=-1
            filter_ignored = tf.not_equal(cls_target, -1.)
            cls_target = tf.boolean_mask(
                cls_target, filter_ignored, name='mask_targets')
            cls_scores = tf.boolean_mask(
                cls_scores, filter_ignored, name='mask_scores')

            cls_target = tf.Print(
                cls_target,
                [cls_target, cls_scores],
                message="TARG, SCORE: ",
                summarize=210
            )

            nonbackground = tf.greater(tf.argmax(cls_scores, axis=1), 0)
            nonbackground_n = tf.shape(tf.where(nonbackground))[0]

            nonbackground_t = tf.greater(cls_target, 0)
            nonbackground_t_n = tf.shape(tf.where(nonbackground_t))[0]
            cls_scores = tf.Print(cls_scores,
                                  [nonbackground_n, nonbackground_t_n],
                                  message='NBK, NBK_T: ')

            bbox_preds = tf.boolean_mask(
                bbox_preds, filter_ignored, name='mask_proposals')
            bbox_target = tf.boolean_mask(
                bbox_target, filter_ignored, name='mask_bbox_targets')

            cls_loss = focal_loss(
                cls_scores, cls_target,
                self._num_classes, gamma=self._gamma,
                weights=self._class_weight,
                background_divider=self._background_divider,
                use_softmax=self._use_softmax
            )
            reg_loss = smooth_l1_loss(
                bbox_preds, bbox_target
            )

            if self._reduce_sum:
                total_cls_loss = tf.reduce_sum(cls_loss)
                total_reg_loss = tf.reduce_sum(reg_loss)
            else:
                total_cls_loss = tf.reduce_mean(cls_loss)
                total_reg_loss = tf.reduce_mean(reg_loss)

            total_reg_loss = tf.Print(
                total_reg_loss,
                [total_cls_loss, total_reg_loss, cls_loss, reg_loss],
                message="CLS_TOT, REG_TOT, CLS, REG, CLS_SH, REG_SH: ",
                summarize=20
            )

            tf.losses.add_loss(total_cls_loss)
            tf.losses.add_loss(total_reg_loss)

            tf.summary.scalar(
                'cls_loss', total_cls_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'reg_loss', total_reg_loss,
                collections=self._losses_collections
            )

            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar(
                'total_loss', total_loss,
                collections=self._losses_collections
            )
            if return_all:
                return {
                    'total_loss': total_loss
                }
            return total_loss

    def _generate_anchors(self, level_shape, im_shape):
        """Generate anchors for a level in the pyramid.
        """
        with tf.variable_scope('generate_anchors'):
            level_height = level_shape[0]
            level_width = level_shape[1]

            im_height = im_shape[0]
            im_width = im_shape[1]

            shift_x = tf.range(level_width, name='range_width')
            shift_y = tf.range(level_height, name='range_height')
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

            shift_x = tf.reshape(shift_x, [-1], name='reshape_x')
            shift_y = tf.reshape(shift_y, [-1], name='reshape_y')

            shifts = tf.stack(
                [shift_x, shift_y, shift_x, shift_y],
                axis=0, name='stack_shifts'
            )

            shifts = tf.transpose(shifts, name='transpose_shifts')
            # Shifts now is a (H x W, 4) Tensor
            # Expand dims to use broadcasting sum.
            level_anchors = (
                np.expand_dims(self._anchor_reference, axis=0) +
                tf.expand_dims(shifts, axis=1, name='expand_shifts')
            )

            # Flatten
            level_anchors = tf.reshape(
                level_anchors, (-1, 4), name='reshape_anchors'
            )
            level_anchors = adjust_bboxes(
                level_anchors,
                old_height=level_height, old_width=level_width,
                new_height=im_height, new_width=im_width
            )
            return level_anchors

    def _add_fpn_levels(self, fpn_levels):
        """Get two more levels from the FPN.

        We compute these levels using a 3x3 Conv layer with stride 2 for the
        first new level and ReLU + another 3x3 Conv layer with stride 2 for the
        second new level.
        """
        with tf.name_scope('add_new_fpn_levels'):
            # TODO: make this more configurable?
            self._conv_new_level_first = snt.Conv2D(
                output_channels=self._config.model.fpn.num_channels,
                kernel_shape=[3, 3], stride=2,
                name='fpn_new_level_first'
            )
            self._conv_new_level_second = snt.Conv2D(
                output_channels=self._config.model.fpn.num_channels,
                kernel_shape=[3, 3], stride=2,
                name='fpn_new_level_second'
            )
            new_level_first = self._conv_new_level_first(
                self.fpn.end_points[0]
            )
            new_level_second = self._conv_new_level_second(
                tf.nn.relu(new_level_first)
            )
            # Maintaining increasing order for consistency.
            all_levels = [new_level_second, new_level_first] + fpn_levels
            return all_levels

    @property
    def summary(self):
        summaries = [tf.summary.merge_all(key=self._losses_collections[0])]
        summaries.append(tf.summary.merge_all(key='retina'))
        return tf.summary.merge(summaries)

    def get_trainable_vars(self):
        # TODO: allow fine_tune_from, etc.
        trainable_vars = snt.get_variables_in_module(self)
        trainable_vars += self.fpn.get_trainable_vars(
            train_base=self._config.model.fpn.train_base
        )
        return trainable_vars

    def get_saver(self):
        return get_saver((self, self.fpn))

    def load_pretrained_weights(self):
        return self.fpn.load_weights()
