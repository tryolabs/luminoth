import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from sonnet.python.modules.conv import Conv2D

class RPN(snt.AbstractModule):
    def __init__(self, anchor_scales, anchor_ratios, num_channels=512,
                 kernel_shape=[3, 3], is_training=False, name='rpn'):
        """RPN"""
        super(RPN, self).__init__(name=name)
        self._anchor_scales = anchor_scales
        self._anchor_ratios = anchor_ratios
        self._num_anchors = len(anchor_scales) * len(anchor_ratios)

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
                initializers={'w': self._initializer}, name='conv'
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

        rpn = self._rpn_activation(
            self._rpn(pretrained, is_training=is_training))

        rpn_cls_score = self._rpn_cls(rpn, is_training=is_training)
        rpn_cls_score_reshape = self._reshape_layer(
            rpn_cls_score, 2, 'rpn_cls_score_reshape')

        # TODO: Malas dimensiones con el reshape de arriba.
        rpn_cls_prob_reshape = tf.nn.softmax(
            rpn_cls_score_reshape, name='rpn_cls_prob_reshape')
        rpn_cls_prob = self._reshape_layer(
            rpn_cls_prob_reshape, self._num_anchors * 2, 'rpn_cls_prob')


        rpn_bbox_pred = self._rpn_bbox(rpn, is_training=is_training)

        return rpn_cls_prob, rpn_bbox_pred


    def _reshape_layer(self, bottom, num_dim, *args):
        # TODO: tf-faster-rcnn/lib/network.py
        raise NotImplemented()
