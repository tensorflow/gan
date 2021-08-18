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
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.data.experimental import AUTOTUNE


def random_flip(lr_img, hr_img):
  """ Randomly flips LR and HR images for data augmentation."""
  rn = tf.random.uniform(shape=(), maxval=1)
  
  return tf.cond(rn < 0.5,
                 lambda: (lr_img, hr_img),
                 lambda: (tf.image.flip_left_right(lr_img),
                          tf.image.flip_left_right(hr_img)))

def random_rotate(lr_img, hr_img):
  """ Randomly rotates LR and HR images for data augmentation."""
  rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
  return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


def get_div2k_data(HParams, 
                   mode='train',
                   shuffle=True, 
                   repeat_count=None):
  """ Downloads and loads DIV2K dataset. 
  Args:
      HParams : For getting values for different parameters.
      mode : Either 'train' or 'valid'.
      shuffle : Whether to shuffle the images in the dataset.
      repeat_count : Repetition of data during training.
  Returns:
      A tf.data.Dataset with pairs of LR image and HR image tensors.

  Raises:
        TypeError : If the data directory(data_dir) is not specified.
  """
  bs = HParams.batch_size
  split = 'train' if mode == 'train' else 'validation'

  def scale(image, *args):
    hr_size = HParams.hr_dimension
    scale = HParams.scale

    hr_image = image
    hr_image = tf.image.resize(hr_image, [hr_size, hr_size])
    lr_image = tf.image.resize(hr_image, [hr_size//scale, hr_size//scale], method='bicubic')
    
    hr_image = tf.clip_by_value(hr_image, 0, 255)
    lr_image = tf.clip_by_value(lr_image, 0, 255)
    
    return lr_image, hr_image

  dataset = (tfds.load('div2k/bicubic_x4', 
                       split=split, 
                       data_dir=HParams.data_dir, 
                       as_supervised=True)
             .map(scale, num_parallel_calls=4)
             .cache())
  
  if shuffle:
    dataset = dataset.shuffle(
        buffer_size=10000, reshuffle_each_iteration=True)
  
  dataset = dataset.batch(HParams.batch_size)
  dataset = dataset.repeat(repeat_count)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
            
  return dataset
