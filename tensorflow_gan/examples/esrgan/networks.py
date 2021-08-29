# coding=utf-8
# Copyright 2021 The TensorFlow GAN Authors.
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

"""Implementation of Generator (ESRGAN_G) and Discriminator (ESRGAN_D) models
   based on the architecture proposed in the paper 'ESRGAN: Enhanced 
   Super-Resolution Generative Adversarial Networks'.
   (https://arxiv.org/abs/1809.00219).
"""

import tensorflow as tf
from tensorflow.keras import layers

def _conv_block(input, filters, activation=True):
  h = layers.Conv2D(filters, kernel_size=[3, 3], 
                    kernel_initializer="he_normal", bias_initializer="zeros", 
                    strides=[1, 1], padding='same', use_bias=True)(input)
  if activation:
      h = layers.LeakyReLU(0.2)(h)
  return h

def _conv_block_d(input, out_channel):
  x = layers.Conv2D(out_channel, 3, 1, padding='same', use_bias=False)(input)
  x = layers.BatchNormalization(momentum=0.8)(x)
  x = layers.LeakyReLU(alpha=0.2)(x)

  x = layers.Conv2D(out_channel, 4, 2, padding='same', use_bias=False)(x)
  x = layers.BatchNormalization(momentum=0.8)(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  return x

def dense_block(input):
  h1 = _conv_block(input, 32)
  h1 = layers.Concatenate()([input, h1])

  h2 = _conv_block(h1, 32)
  h2 = layers.Concatenate()([input, h1, h2])

  h3 = _conv_block(h2, 32)
  h3 = layers.Concatenate()([input, h1, h2, h3])

  h4 = _conv_block(h3, 32)
  h4 = layers.Concatenate()([input, h1, h2, h3, h4])  

  h5 = _conv_block(h4, 32, activation=False)
  
  h5 = layers.Lambda(lambda x: x * 0.2)(h5)
  h = layers.Add()([h5, input])
  
  return h

def rrdb(input):
  h = dense_block(input)
  h = dense_block(h)
  h = dense_block(h)
  h = layers.Lambda(lambda x: x * 0.2)(h)
  out = layers.Add()([h, input])
  return out

def upsample(x, filters):
  x = layers.Conv2DTranspose(filters, kernel_size=3, 
                             strides=2, padding='same', 
                             use_bias=True)(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  return x

def generator_network(hparams,
                      num_filters=32,
                      out_channels=3):
  """The Generator network for ESRGAN consisting of Residual in Residual 
     Block as the basic building unit.

  Args :
      num_filters : Number of num_filters for the convolutional layers used.
      out_channels : Number of channels for the generated image.
      use_bias : Whether to use bias or not for the convolutional layers.

  Returns:
      The compiled model of the generator network where the inputs and outputs
      of the model are defined as :
          inputs -> Batch of tensors representing LR images.
          outputs -> Batch of generated HR images.
  """
  lr_input = layers.Input(shape=(hparams.hr_dimension//hparams.scale,
                                 hparams.hr_dimension//hparams.scale, 3))

  x = layers.Conv2D(num_filters, kernel_size=[3, 3], strides=[1, 1],
                    padding='same', use_bias=True)(lr_input)
  x = layers.LeakyReLU(0.2)(x)

  ref = x

  for _ in range(hparams.trunk_size):
    x = rrdb(x)

  x = layers.Conv2D(num_filters, kernel_size=[3, 3], strides=[1, 1],
                    padding='same', use_bias=True)(x)
  x = layers.Add()([x, ref])

  x = upsample(x, num_filters)
  x = upsample(x, num_filters)

  x = layers.Conv2D(num_filters, kernel_size=[3, 3], strides=[1, 1],
                    padding='same', use_bias=True)(x)
  x = layers.LeakyReLU(0.2)(x)
  hr_output = layers.Conv2D(out_channels, kernel_size=[3, 3], strides=[1, 1],
                            padding='same', use_bias=True)(x)

  model = tf.keras.models.Model(inputs=lr_input, outputs=hr_output)
  return model


def discriminator_network(hparams,
                          num_filters=64, 
                          training=True):
  """The discriminator network for ESRGAN.

  Args :
      num_filters : Number of filters for the first convolutional layer.
  Returns :
      The compiled model of the discriminator network where the inputs
      and outputs of the model are defined as :
          inputs -> Batch of tensors representing HR images.
          outputs -> Predictions for batch of input images.
  """
  img = layers.Input(shape=(hparams.hr_dimension, hparams.hr_dimension, 3))

  x = layers.Conv2D(num_filters, [3, 3], 1, padding='same', use_bias=False)(img)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU(alpha=0.2)(x)

  x = layers.Conv2D(num_filters, [3, 3], 2, padding='same', use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU(alpha=0.2)(x)

  x = _conv_block_d(x, num_filters *2)
  x = _conv_block_d(x, num_filters *4)
  x = _conv_block_d(x, num_filters *8)
  
  x = layers.Flatten()(x)
  x = layers.Dense(100)(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Dense(1)(x)

  model = tf.keras.models.Model(inputs=img, outputs=x)
  return model
