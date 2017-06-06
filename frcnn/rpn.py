import collections
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from sonnet.python.modules.conv import Conv2D
from utils.generate_anchors import generate_anchors


class RPN(snt.AbstractModule):
    def __init__(self, anchor_scales, anchor_ratios, num_channels=512,
                 kernel_shape=[3, 3], is_training=False, name='rpn'):
        """RPN"""
        super(RPN, self).__init__(name=name)

        if not isinstance(anchor_scales, collections.Iterable):
            raise TypeError("anchor_scales must be iterable")
        anchor_scales = tuple(anchor_scales)

        if not isinstance(anchor_ratios, collections.Iterable):
            raise TypeError("anchor_ratios must be iterable")
        anchor_ratios = tuple(anchor_ratios)

        if not isinstance(kernel_shape, collections.Iterable):
            raise TypeError("kernel_shape must be iterable")
        kernel_shape = tuple(kernel_shape)

        if not anchor_scales:
            raise ValueError("anchor_scales must not be empty")
        if not anchor_ratios:
            raise ValueError("anchor_ratios must not be empty")

        self._anchor_scales = anchor_scales
        self._anchor_ratios = anchor_ratios
        # TODO: Do we need the anchors? Can't we just use len(self._anchor_scales) * len(self._anchor_ratios)
        self._num_anchors = self.anchors.shape[0]

        print(self.anchors.shape)

        self._num_channels = num_channels
        self._kernel_shape = kernel_shape

        self._initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self._rpn_activation = tf.nn.relu

        self._instantiate_layers()

    def _instantiate_layers(self):
        """Instantiates all convolutional modules used in the RPN."""
        with self._enter_variable_scope():
            self._rpn = Conv2D(
                output_channels=self._num_channels,
                kernel_shape=self._kernel_shape,
                initializers={'w': self._initializer, 'b': self._initializer}, name='conv'
            )

            self._rpn_cls = Conv2D(
                output_channels=self._num_anchors * 2, kernel_shape=[1, 1],
                initializers={'w': self._initializer}, padding='VALID',
                name='cls_conv'
            )

            # BBox prediction is 4 values * number of anchors.
            self._rpn_bbox = Conv2D(
                output_channels=self._num_anchors * 4, kernel_shape=[1, 1],
                initializers={'w': self._initializer}, padding='VALID',
                name='bbox_conv'
            )

    def _build(self, pretrained, is_training=True):
        """
        TODO: We don't have BatchNorm yet.
        """
        rpn = self._rpn_activation(
            self._rpn(pretrained))

        rpn_cls_score = self._rpn_cls(rpn)
        rpn_cls_score_reshape = self.spatial_reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = self.spatial_softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.spatial_reshape_layer(rpn_cls_prob, self._num_anchors * 2)
        rpn_bbox_pred = self._rpn_bbox(rpn)

        return {
            'rpn': rpn,
            'rpn_cls_prob': rpn_cls_prob,
            'rpn_cls_prob_reshape': rpn_cls_prob_reshape,
            'rpn_cls_score': rpn_cls_score,
            'rpn_cls_score_reshape': rpn_cls_score_reshape,
            'rpn_bbox_pred': rpn_bbox_pred,
        }

    def spatial_softmax(self, input):
        input_shape = tf.shape(input)
        reshaped_input = tf.reshape(input, [-1, input_shape[3]])
        softmaxed = tf.nn.softmax(reshaped_input)
        return tf.reshape(
            softmaxed, [-1, input_shape[1], input_shape[2], input_shape[3]]
        )

    def spatial_reshape_layer(self, input, num_dim):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(
            input, [
                input_shape[0],
                input_shape[1],
                -1,
                int(num_dim)
        ])

    @property
    def anchors(self):
        if not hasattr(self, '_anchors') or self._anchors is None:
            self._anchors = self._generate_anchors()
        return self._anchors

    def _generate_anchors(self):
        """
        Generate anchors based on the ratios and scales.

        We use the original code found in `rbgirshick/py-faster-rcnn`
        """
        return generate_anchors(
            ratios=np.array(self._anchor_ratios), scales=np.array(self._anchor_scales)
        )
