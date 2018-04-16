# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# Taken from tensorboard repository:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py
# Modified to remove the fully connected layers from vgg16.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope


def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected],
        activation_fn=nn_ops.relu,
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        biases_initializer=init_ops.zeros_initializer()
    ):
        with arg_scope([layers.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def truncated_vgg_16(inputs, is_training=True, scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.

    For use in SSD object detection network, which has this particular
    truncated version of VGG16 detailed in its paper.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      scope: Optional scope for the variables.

    Returns:
      the last op containing the conv5 tensor and end_points dict.
    """
    with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with arg_scope(
            [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
            outputs_collections=end_points_collection
        ):
            net = layers_lib.repeat(
                inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
            net = layers_lib.repeat(
                net, 2, layers.conv2d, 128, [3, 3], scope='conv2'
            )
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
            net = layers_lib.repeat(
                net, 3, layers.conv2d, 256, [3, 3], scope='conv3'
            )
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
            net = layers_lib.repeat(
                net, 3, layers.conv2d, 512, [3, 3], scope='conv4'
            )
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
            net = layers_lib.repeat(
                net, 3, layers.conv2d, 512, [3, 3], scope='conv5'
            )
            # Convert end_points_collection into a end_point dict.
            end_points = utils.convert_collection_to_dict(
                end_points_collection
            )
            return net, end_points


truncated_vgg_16.default_image_size = 224
