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

"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
  return tf.compat.v1.layers.batch_normalization(
      x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)


def _dense(x, channels, name):
  return tf.compat.v1.layers.dense(
      x,
      channels,
      kernel_initializer=tf.compat.v1.initializers.truncated_normal(
          stddev=0.02),
      name=name)


def _conv2d(x, filters, kernel_size, stride, name):
  return tf.compat.v1.layers.conv2d(
      x,
      filters, [kernel_size, kernel_size],
      strides=[stride, stride],
      padding='same',
      kernel_initializer=tf.compat.v1.initializers.truncated_normal(
          stddev=0.02),
      name=name)


def _deconv2d(x, filters, kernel_size, stride, name):
  return tf.compat.v1.layers.conv2d_transpose(
      x,
      filters, [kernel_size, kernel_size],
      strides=[stride, stride],
      padding='same',
      kernel_initializer=tf.compat.v1.initializers.truncated_normal(
          stddev=0.02),
      name=name)


def discriminator(images, unused_conditioning, is_training=True,
                  scope='Discriminator'):
  """Discriminator for CIFAR images.

  Args:
    images: A Tensor of shape [batch size, width, height, channels], that can be
      either real or generated. It is the discriminator's goal to distinguish
      between the two.
    unused_conditioning: The TFGAN API can help with conditional GANs, which
      would require extra `condition` information to both the generator and the
      discriminator. Since this example is not conditional, we do not use this
      argument.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.
    scope: A variable scope or string for the discriminator.

  Returns:
    A 1D Tensor of shape [batch size] representing the confidence that the
    images are real. The output can lie in [-inf, inf], with positive values
    indicating high confidence that the images are real.
  """
  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    x = _conv2d(images, 64, 5, 2, name='d_conv1')
    x = _leaky_relu(x)

    x = _conv2d(x, 128, 5, 2, name='d_conv2')
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))

    x = _conv2d(x, 256, 5, 2, name='d_conv3')
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn3'))

    x = tf.reshape(x, [-1, 4 * 4 * 256])

    x = _dense(x, 1, name='d_fc_4')

    return x


def generator(noise, is_training=True, scope='Generator'):
  """Generator to produce CIFAR images.

  Args:
    noise: A 2D Tensor of shape [batch size, noise dim]. Since this example
      does not use conditioning, this Tensor represents a noise vector of some
      kind that will be reshaped by the generator into CIFAR examples.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.
    scope: A variable scope or string for the generator.

  Returns:
    A single Tensor with a batch of generated CIFAR images.
  """
  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    net = _dense(noise, 4096, name='g_fc1')
    net = tf.nn.relu(_batch_norm(net, is_training, name='g_bn1'))

    net = tf.reshape(net, [-1, 4, 4, 256])

    net = _deconv2d(net, 128, 5, 2, name='g_dconv2')
    net = tf.nn.relu(_batch_norm(net, is_training, name='g_bn2'))

    net = _deconv2d(net, 64, 4, 2, name='g_dconv3')
    net = tf.nn.relu(_batch_norm(net, is_training, name='g_bn3'))

    net = _deconv2d(net, 3, 4, 2, name='g_dconv4')
    net = tf.tanh(net)

    return net
