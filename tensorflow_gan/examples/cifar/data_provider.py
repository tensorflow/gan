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

"""Contains code for loading and preprocessing the CIFAR data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds


def provide_dataset(split,
                    batch_size,
                    num_parallel_calls=None,
                    shuffle=True,
                    one_hot=True):
  """Provides batches of CIFAR10 digits.

  Args:
    split: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.
    one_hot: Output one hot vector instead of int32 label.

  Returns:
    A tf.data.Dataset with:
      * images: A `Tensor` of size [batch_size, 32, 32, 3] and type tf.float32.
          Output pixel values are in [-1, 1].
      * labels: A `Tensor` of size [batch_size, 10] of one-hot label
          encodings with type tf.int32, or a `Tensor` of size [batch_size],
          depending on the value of `one_hot`.

  Raises:
    ValueError: If `split` isn't `train` or `test`.
  """
  ds = tfds.load('cifar10', split=split, shuffle_files=shuffle)

  def _preprocess(element):
    """Map elements to the example dicts expected by the model."""
    images = (tf.cast(element['image'], tf.float32) - 128.0) / 128.0
    labels = element['label']
    if one_hot:
      num_classes = 10
      labels = tf.one_hot(labels, num_classes)
    else:
      labels = tf.cast(labels, tf.float32)
    return {'images': images, 'labels': labels}

  ds = (
      ds.map(_preprocess,
             num_parallel_calls=num_parallel_calls).cache().repeat())
  if shuffle:
    ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
  ds = (
      ds.batch(batch_size,
               drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

  return ds


def provide_data(split,
                 batch_size,
                 num_parallel_calls=None,
                 shuffle=True,
                 one_hot=True):
  """Provides batches of CIFAR10 digits.

  Args:
    split: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.
    one_hot: Output one hot vector instead of int32 label.

  Returns:
    images: A `Tensor` of size [batch_size, 32, 32, 3]. Output pixel values are
      in [-1, 1].
    labels: A `Tensor` of size [batch_size, 10] of one-hot label
      encodings with type tf.int32, or a `Tensor` of size [batch_size],
      depending on the value of `one_hot`.

  Raises:
    ValueError: If `split` isn't `train` or `test`.
  """
  ds = provide_dataset(split, batch_size, num_parallel_calls, shuffle, one_hot)

  next_batch = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
  images, labels = next_batch['images'], next_batch['labels']

  return images, labels


def float_image_to_uint8(image):
  """Convert float image in [-1, 1) to [0, 255] uint8.

  Note that `1` gets mapped to `0`, but `1 - epsilon` gets mapped to 255.

  Args:
    image: An image tensor. Values should be in [-1, 1).

  Returns:
    Input image cast to uint8 and with integer values in [0, 255].
  """
  image = (image * 128.0) + 128.0
  return tf.cast(image, tf.uint8)
