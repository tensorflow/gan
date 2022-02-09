# coding=utf-8
# Copyright 2022 The TensorFlow GAN Authors.
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

"""Implementation of the generator and discriminator networks.

Based on the architecture proposed in the paper 'ESRGAN: Enhanced
Super-Resolution Generative Adversarial Networks'.
(https://arxiv.org/abs/1809.00219)
"""

import tensorflow as tf


def _conv_block(x, filters, activation=True):
  """Convolutional block used for building generator network."""
  h = tf.keras.layers.Conv2D(
      filters,
      kernel_size=[3, 3],
      kernel_initializer='he_normal',
      bias_initializer='zeros',
      strides=[1, 1],
      padding='same',
      use_bias=True)(
          x)
  if activation:
    h = tf.keras.layers.LeakyReLU(0.2)(h)
  return h


def _conv_block_d(x, out_channel):
  """Convolutional block used for building discriminator network."""
  x = tf.keras.layers.Conv2D(
      out_channel, kernel_size=3, strides=1, padding='same', use_bias=False)(
          x)
  x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

  x = tf.keras.layers.Conv2D(
      out_channel, kernel_size=4, strides=2, padding='same', use_bias=False)(
          x)
  x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  return x


def dense_block(x):
  """Dense block used inside RRDB."""
  h1 = _conv_block(x, 32)
  h1 = tf.keras.layers.Concatenate()([x, h1])

  h2 = _conv_block(h1, 32)
  h2 = tf.keras.layers.Concatenate()([x, h1, h2])

  h3 = _conv_block(h2, 32)
  h3 = tf.keras.layers.Concatenate()([x, h1, h2, h3])

  h4 = _conv_block(h3, 32)
  h4 = tf.keras.layers.Concatenate()([x, h1, h2, h3, h4])

  h5 = _conv_block(h4, 32, activation=False)

  h5 = tf.keras.layers.Lambda(lambda x: x * 0.2)(h5)
  h = tf.keras.layers.Add()([h5, x])

  return h


def rrdb(x):
  """Residual-in-Residual Dense Block used in the generator."""
  h = dense_block(x)
  h = dense_block(h)
  h = dense_block(h)
  h = tf.keras.layers.Lambda(lambda x: x * 0.2)(h)
  out = tf.keras.layers.Add()([h, x])
  return out


def upsample(x, filters):
  """Upsampling layer for the generator."""
  x = tf.keras.layers.Conv2DTranspose(
      filters, kernel_size=3, strides=2, padding='same', use_bias=True)(
          x)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  return x


def generator_network(hparams, num_filters=32, out_channels=3):
  """The generator network for the ESRGAN model.

  Args:
      hparams: Hyperarameters for network.
      num_filters : Number of num_filters for the convolutional layers used.
      out_channels : Number of channels for the generated image.

  Returns:
      The compiled model of the generator network where the inputs and outputs
      of the model are defined as :
          inputs -> Batch of tensors representing LR images.
          outputs -> Batch of generated HR images.
  """
  lr_input = tf.keras.layers.Input((hparams.hr_dimension // hparams.scale,
                                    hparams.hr_dimension // hparams.scale, 3))

  x = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=[3, 3],
      strides=[1, 1],
      padding='same',
      use_bias=True)(
          lr_input)
  x = tf.keras.layers.LeakyReLU(0.2)(x)

  ref = x

  for _ in range(hparams.trunk_size):
    x = rrdb(x)

  x = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=[3, 3],
      strides=[1, 1],
      padding='same',
      use_bias=True)(
          x)
  x = tf.keras.layers.Add()([x, ref])

  x = upsample(x, num_filters)
  x = upsample(x, num_filters)

  x = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=[3, 3],
      strides=[1, 1],
      padding='same',
      use_bias=True)(
          x)
  x = tf.keras.layers.LeakyReLU(0.2)(x)
  hr_output = tf.keras.layers.Conv2D(
      out_channels,
      kernel_size=[3, 3],
      strides=[1, 1],
      padding='same',
      use_bias=True)(
          x)

  model = tf.keras.models.Model(inputs=lr_input, outputs=hr_output)
  return model


def discriminator_network(hparams, num_filters=64):
  """The discriminator network for the ESRGAN model.

  Args:
      hparams: Parameters for the network.
      num_filters : Number of filters for the first convolutional layer.

  Returns:
      The compiled model of the discriminator network where the inputs
      and outputs of the model are defined as :
          inputs -> Batch of tensors representing HR images.
          outputs -> Predictions for batch of input images.
  """
  img = tf.keras.layers.Input(
      shape=(hparams.hr_dimension, hparams.hr_dimension, 3))

  x = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=[3, 3],
      strides=1,
      padding='same',
      use_bias=False)(
          img)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

  x = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=[3, 3],
      strides=2,
      padding='same',
      use_bias=False)(
          x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

  x = _conv_block_d(x, num_filters * 2)
  x = _conv_block_d(x, num_filters * 4)
  x = _conv_block_d(x, num_filters * 8)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(100)(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  x = tf.keras.layers.Dense(1)(x)

  model = tf.keras.models.Model(inputs=img, outputs=x)
  return model
