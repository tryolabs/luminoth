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
from luminoth.utils.config import get_base_config
from luminoth.utils.image import adjust_bboxes
from luminoth.utils.losses import smooth_l1_loss, focal_loss
from luminoth.utils.vars import get_saver


class Retina(snt.AbstractModule):

    base_config = get_base_config(__file__)

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

        self._gamma = self._config.model.loss.gamma

        self._offset = self._config.model.anchors.offset

    def _build(self, image, gt_boxes=None, is_training=True):
        im_shape = tf.to_float(tf.shape(image)[1:3])
        fpn_levels = self.fpn(image, is_training=is_training)

        # Add new FPN levels (AMIP)
        fpn_levels = self._add_fpn_levels(fpn_levels)

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

        proposals = []
        class_scores = []
        class_probs = []
        all_anchors = []
        for level in fpn_levels:
            level_shape_int = tf.shape(level)[1:3]
            level_shape = tf.to_float(level_shape_int)
            anchors = self._generate_anchors(tf.shape(level))
            all_anchors.append(anchors)
            level_bbox_pred_bank = box_subnet(level, anchors)
            level_bbox_pred = tf.reshape(
                level_bbox_pred_bank,
                shape=[-1, 4]
            )
            level_class_score_bank = class_subnet(level, anchors)
            level_class_scores = tf.reshape(
                level_class_score_bank,
                shape=[-1, self._num_classes + 1]
            )
            level_class_probs = tf.nn.softmax(level_class_scores)

            level_proposals = decode(anchors, level_bbox_pred)

            # Get rid of proposals from anchors outside the image.
            xmin, ymin, xmax, ymax = tf.split(anchors, 4, axis=1)
            proposals_filter = tf.logical_and(
                tf.logical_and(
                    tf.greater_equal(xmin, 0 - self._offset),
                    tf.greater_equal(ymin, 0 - self._offset),
                ),
                tf.logical_and(
                    tf.less_equal(xmax, level_shape_int[0] + self._offset),
                    tf.less_equal(ymax, level_shape_int[1] + self._offset),
                ),
            )
            proposals_filter = tf.reshape(proposals_filter, [-1])
            level_proposals = tf.boolean_mask(
                level_proposals, proposals_filter, name='mask_proposals'
            )
            level_class_scores = tf.boolean_mask(
                level_class_scores, proposals_filter, name='mask_scores'
            )
            level_class_probs = tf.boolean_mask(
                level_class_probs, proposals_filter, name='mask_probs'
            )

            # Now resize proposals to be relative to the full image size
            # instead of relative to the level size.
            level_proposals = tf.to_float(adjust_bboxes(
                level_proposals,
                old_height=level_shape[0], old_width=level_shape[1],
                new_height=im_shape[0], new_width=im_shape[1]
            ))
            proposals.append(level_proposals)
            class_scores.append(level_class_scores)
            class_probs.append(level_class_probs)

        class_scores = tf.concat(class_scores, axis=0)
        class_probs = tf.concat(class_probs, axis=0)

        proposals = tf.concat(proposals, axis=0)
        all_anchors = tf.concat(all_anchors, axis=0)
        proposal_dict = self._proposal(
            class_probs, class_scores, proposals, all_anchors, im_shape
        )

        class_probs = proposal_dict['proposal_label_prob']
        proposals = proposal_dict['objects']

        pred_dict = {
            'cls_scores': proposal_dict['proposal_label_score'],
            'cls_probs': proposal_dict['proposal_label_prob'],
            'proposals': proposal_dict['objects'],
        }

        if gt_boxes is not None:
            gt_boxes = tf.to_float(gt_boxes)
            retina_target = RetinaTarget(
                self._config.model.target, num_classes=self._num_classes,
            )
            pred_dict['cls_target'], pred_dict['bbox_target'] = retina_target(
                pred_dict['proposals'], gt_boxes
            )
        return pred_dict

    def loss(self, pred_dict):
        """Compute training loss for object detection with Retina.

        We use focal loss for classification, and smooth L1 loss for bbox
        regression.
        """
        with tf.name_scope('losses'):
            cls_probs = pred_dict['cls_probs']
            cls_scores = pred_dict['cls_scores']
            cls_target = pred_dict['cls_target']

            bbox_pred = pred_dict['proposals']
            bbox_target = pred_dict['bbox_target']

            cls_loss = focal_loss(
                cls_scores, cls_probs, cls_target,
                self._num_classes, gamma=self._gamma,
                # TODO: implement this!!!!
                weights=None
            )
            reg_loss = smooth_l1_loss(
                bbox_pred, bbox_target
            )

            total_cls_loss = tf.reduce_sum(cls_loss)
            total_reg_loss = tf.reduce_sum(reg_loss)

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
            return total_loss

    def _generate_anchors(self, level_shape):
        """Generate anchors for a level in the pyramid.
        """
        with tf.variable_scope('generate_anchors'):
            level_height = level_shape[1]
            level_width = level_shape[2]

            shift_x = tf.range(level_width)
            shift_y = tf.range(level_height)
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
            level_anchors = (
                np.expand_dims(self._anchor_reference, axis=0) +
                tf.expand_dims(shifts, axis=1)
            )

            # Flatten
            level_anchors = tf.reshape(
                level_anchors, (-1, 4)
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
                fpn_levels[0]
            )
            new_level_second = self._conv_new_level_second(
                tf.nn.relu(new_level_first)
            )
            # Maintaining increasing order for consistency.
            return [new_level_second, new_level_first] + fpn_levels

    @property
    def summary(self):
        summaries = [tf.summary.merge_all(key=self._losses_collections[0])]
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
