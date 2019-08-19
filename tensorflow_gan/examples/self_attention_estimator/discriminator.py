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

"""Discriminator definitions."""

import tensorflow as tf
from tensorflow_gan.examples import compat_utils
from tensorflow_gan.examples.self_attention_estimator import ops


def dsample(x):
  """Downsamples the input volume by means of average pooling.

  Args:
    x: The 4D input tensor.
  Returns:
    An downsampled version of the input tensor.
  """
  xd = compat_utils.nn_avg_pool2d(
      input=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
  return xd


def block(x, out_channels, name, downsample=True, act=tf.nn.relu):
  """Builds the residual blocks used in the discriminator.

  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    downsample: If True, downsample the spatial size the input tensor by
                a factor of 2 on each side. If False, the spatial size of the
                input tensor is unchanged.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.compat.v1.variable_scope(name):
    input_channels = x.shape.as_list()[-1]
    x_0 = x
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='sn_conv1')
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='sn_conv2')
    if downsample:
      x = dsample(x)
    if downsample or input_channels != out_channels:
      x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, name='sn_conv3')
      if downsample:
        x_0 = dsample(x_0)
    return x_0 + x


def optimized_block(x, out_channels, name, act=tf.nn.relu):
  """Builds optimized residual blocks for downsampling.

  Compared with block, optimized_block always downsamples the spatial resolution
  by a factor of 2 on each side.

  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.compat.v1.variable_scope(name):
    x_0 = x
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='sn_conv1')
    x = act(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='sn_conv2')
    x = dsample(x)
    x_0 = dsample(x_0)
    x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, name='sn_conv3')
    return x + x_0


def discriminator(image, labels, df_dim, number_classes, act=tf.nn.relu):
  """Builds the discriminator graph.

  Args:
    image: The current batch of images to classify as fake or real.
    labels: The corresponding labels for the images.
    df_dim: The df dimension.
    number_classes: The number of classes in the labels.
    act: The activation function used in the discriminator.
  Returns:
    - A `Tensor` representing the logits of the discriminator.
    - A list containing all trainable varaibles defined by the model.
  """
  with tf.compat.v1.variable_scope(
      'discriminator', reuse=tf.compat.v1.AUTO_REUSE) as dis_scope:
    h0 = optimized_block(
        image, df_dim, 'd_optimized_block1', act=act)  # 64 * 64
    h1 = block(h0, df_dim * 2, 'd_block2', act=act)  # 32 * 32
    h1 = ops.sn_non_local_block_sim(h1, name='d_ops')  # 32 * 32
    h2 = block(h1, df_dim * 4, 'd_block3', act=act)  # 16 * 16
    h3 = block(h2, df_dim * 8, 'd_block4', act=act)  # 8 * 8
    h4 = block(h3, df_dim * 16, 'd_block5', act=act)  # 4 * 4
    h5 = block(h4, df_dim * 16, 'd_block6', downsample=False, act=act)
    h5_act = act(h5)
    h6 = tf.reduce_sum(input_tensor=h5_act, axis=[1, 2])
    output = ops.snlinear(h6, 1, name='d_sn_linear')
    h_labels = ops.sn_embedding(labels, number_classes, df_dim * 16,
                                name='d_embedding')
    output += tf.reduce_sum(input_tensor=h6 * h_labels, axis=1, keepdims=True)
  var_list = tf.compat.v1.get_collection(
      tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, dis_scope.name)
  return output, var_list
