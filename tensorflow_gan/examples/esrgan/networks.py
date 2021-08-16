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

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
from keras.models import Model
from keras.layers import Input, Add, Concatenate
from keras.layers import BatchNormalization, LeakyReLU, Conv2D, Dense, Conv2DTranspose
from keras.layers import Lambda, Dropout, Flatten


"""
Implementation of Generator (ESRGAN_G) and Discriminator (ESRGAN_D) models
based on the architecture proposed in the paper
'ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks'.
"""

def _conv_block(input, filters=32, activation=True):
  h = Conv2D(filters, kernel_size=[3,3], bias_initializer="zeros", 
             strides=[1,1], padding='same')(input)
  if activation:
    h = LeakyReLU(0.2)(h)
  return h

def _conv_block_d(input, out_channel):
  x = Conv2D(out_channel, 3,1, padding='same', use_bias=False)(input)
  x = BatchNormalization(momentum=0.8)(x)
  x = LeakyReLU(alpha=0.2)(x)

  x = Conv2D(out_channel, 4,2, padding='same', use_bias=False)(x)
  x = BatchNormalization(momentum=0.8)(x)
  x = LeakyReLU(alpha=0.2)(x)
  return x

def dense_block(input):
  h1 = _conv_block(input)
  h1 = Concatenate()([input, h1])

  h2 = _conv_block(h1)
  h2 = Concatenate()([input, h1, h2])

  h3 = _conv_block(h2)
  h3 = Concatenate()([input, h1, h2, h3])

  h4 = _conv_block(h3)
  h4 = Concatenate()([input, h1, h2, h3, h4])

  h5 = _conv_block(h4, activation=False)

  h5 = Lambda(lambda x: x * 0.2)(h5)
  h = Add()([h5, input])

  return h

def RRDB(input):
  h = dense_block(input)
  h = dense_block(h)
  h = dense_block(h)
  h = Lambda(lambda x: x * 0.2)(h)
  out = Add()([h, input])
  return out

def upsample(x, filters, use_bias=True):
  x = Conv2DTranspose(filters, kernel_size=[3, 3],
                      strides=[2, 2], padding='same',
                      use_bias=use_bias)(x)
  x = LeakyReLU(alpha=0.2)(x)
  return x

def ESRGAN_G(HParams,
             num_filters=32,
             out_channels=3):
  """
  The Generator network for ESRGAN consisting of Residual in Residual Block
  as the basic building unit.

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
  lr_input = Input(shape=(HParams.hr_dimension//HParams.scale,
                          HParams.hr_dimension//HParams.scale, 3))

  x = Conv2D(num_filters, kernel_size=[3, 3], strides=[1, 1],
             padding='same', use_bias=True)(lr_input)
  x = LeakyReLU(0.2)(x)

  ref = x

  for _ in range(HParams.trunk_size):
    x = RRDB(x)

  x = Conv2D(num_filters, kernel_size=[3, 3], strides=[1, 1],
             padding='same', use_bias=True)(x)
  x = Add()([x, ref])

  x = upsample(x, num_filters)
  x = upsample(x, num_filters)

  x = Conv2D(num_filters, kernel_size=[3, 3], strides=[1, 1],
             padding='same', use_bias=True)(x)
  x = LeakyReLU(0.2)(x)
  hr_output = Conv2D(out_channels, kernel_size=[3, 3], strides=[1, 1],
                     padding='same', use_bias=True)(x)

  model = Model(inputs=lr_input, outputs=hr_output)
  return model


def ESRGAN_D(HParams,
             num_filters=64, 
             training=True):
  """
  The discriminator network for ESRGAN.

  Args :
      num_filters : Number of filters for the first convolutional layer.
  Returns :
      The compiled model of the discriminator network where the inputs
      and outputs of the model are defined as :
          inputs -> Batch of tensors representing HR images.
          outputs -> Predictions for batch of input images.
  """
  img = Input(shape=(HParams.hr_dimension, HParams.hr_dimension, 3))

  x = Conv2D(num_filters, [3,3], 1, padding='same', use_bias=False)(img)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.2)(x)

  x = Conv2D(num_filters, [3,3], 2, padding='same', use_bias=False)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.2)(x)

  x = _conv_block_d(x, num_filters *2)
  x = _conv_block_d(x, num_filters *4)
  x = _conv_block_d(x, num_filters *8)
  
  x = Flatten()(x)
  x = Dense(100)(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = Dense(1)(x)

  model = Model(inputs=img, outputs=x)
  return model
