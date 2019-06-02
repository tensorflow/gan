# coding=utf-8
# Copyright 2019 The TensorFlow GAN Authors.
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

"""Implementation of the Image-to-Image Translation model.

This network represents a port of the following work:

  Image-to-Image Translation with Conditional Adversarial Networks
  Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros
  Arxiv, 2017
  https://phillipi.github.io/pix2pix/

A reference implementation written in Lua can be found at:
https://github.com/phillipi/pix2pix/blob/master/models.lua

This code was branched and made TF 2.0 compliant from:
https://github.com/tensorflow/models/blob/master/research/slim/nets/pix2pix.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_gan as tfgan


def _instance_norm(x):
  # These parameters come from the online port, which don't necessarily match
  # those in the paper.
  # TODO(nsilberman): confirm these values with Philip.
  return tfgan.features.instance_norm(
      x,
      center=True,
      scale=True,
      epsilon=0.00001)


def _conv2d(net, num_filters, stride, use_bias, name):
  return tf.compat.v1.layers.conv2d(
      net,
      num_filters,
      kernel_size=[4, 4],
      strides=stride,
      padding='valid',
      use_bias=use_bias,
      kernel_initializer=tf.compat.v1.random_normal_initializer(0, 0.02),
      name=name)


def pix2pix_discriminator(net, num_filters, padding=2, pad_mode='REFLECT',
                          activation_fn=tf.nn.leaky_relu, is_training=False):
  """Creates the Image2Image Translation Discriminator.

  Args:
    net: A `Tensor` of size [batch_size, height, width, channels] representing
      the input.
    num_filters: A list of the filters in the discriminator. The length of the
      list determines the number of layers in the discriminator.
    padding: Amount of reflection padding applied before each convolution.
    pad_mode: mode for tf.pad, one of "CONSTANT", "REFLECT", or "SYMMETRIC".
    activation_fn: activation fn for conv2d.
    is_training: Whether or not the model is training or testing.

  Returns:
    A logits `Tensor` of size [batch_size, N, N, 1] where N is the number of
    'patches' we're attempting to discriminate and a dictionary of model end
    points.
  """
  del is_training
  end_points = {}

  num_layers = len(num_filters)

  def padded(net, scope):
    if padding:
      with tf.compat.v1.variable_scope(scope):
        spatial_pad = tf.constant(
            [[0, 0], [padding, padding], [padding, padding], [0, 0]],
            dtype=tf.int32)
        return tf.pad(tensor=net, paddings=spatial_pad, mode=pad_mode)
    else:
      return net

  # No normalization on the input layer.
  net = padded(net, 'conv0')
  net = _conv2d(net, num_filters[0], stride=2, use_bias=True, name='conv0')
  net = activation_fn(net)
  end_points['conv0'] = net

  for i in range(1, num_layers - 1):
    net = padded(net, 'conv%d' % i)
    net = _conv2d(net, num_filters[i], stride=2, use_bias=False,
                  name='conv%d' % i)
    net = _instance_norm(net)
    net = activation_fn(net)
    end_points['conv%d' % i] = net

  # Stride 1 on the last layer.
  net = padded(net, 'conv%d' % (num_layers - 1))
  net = _conv2d(
      net,
      num_filters[-1],
      stride=1,
      use_bias=False,
      name='conv%d' % (num_layers - 1))
  net = _instance_norm(net)
  net = activation_fn(net)
  end_points['conv%d' % (num_layers - 1)] = net

  # 1-dim logits, stride 1, no activation, no normalization.
  net = padded(net, 'conv%d' % num_layers)
  logits = _conv2d(
      net,
      num_filters=1,
      stride=1,
      use_bias=True,
      name='conv%d' % num_layers)
  end_points['logits'] = logits
  end_points['predictions'] = tf.sigmoid(logits)
  return logits, end_points
