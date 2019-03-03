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

"""Contains code for loading and preprocessing the ImageNet data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds


def provide_dataset(split,
                    batch_size,
                    patch_size,
                    num_parallel_calls=None,
                    shuffle=True):
  """Provides batches of ImageNet digits.

  Args:
    split: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    patch_size: Size of the patch to extract from the image.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.

  Returns:
    A tf.data.Dataset with:
      * images: A `Tensor` of size [batch_size, patch_size, patch_size, 3] and
          type tf.float32. Output pixel values are in (-1, 1).
      * labels: A `Tensor` of size [batch_size] with type tf.float32.

  Raises:
    ValueError: If `split` isn't `train` or `validation`.
  """
  ds = tfds.load('imagenet2012', split=split)

  def _preprocess(element):
    """Map elements to the example dicts expected by the model."""
    images = tf.cast(element['image'], tf.float32)
    labels = tf.cast(element['label'], tf.float32)

    # Preprocess the images. Make the range lie in a strictly smaller range than
    # [-1, 1], so that network outputs aren't forced to the extreme ranges.
    images = (images - 128.0) / 142.0

    patches = tf.image.resize_image_with_crop_or_pad(images, patch_size,
                                                     patch_size)

    patches.shape.assert_is_compatible_with([patch_size, patch_size, 3])
    patches.set_shape([patch_size, patch_size, 3])
    labels.shape.assert_is_compatible_with([])

    return {'images': patches, 'labels': labels}

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
                 patch_size,
                 num_parallel_calls=None,
                 shuffle=True):
  """Provides batches of ImageNet images.

  Args:
    split: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    patch_size: Size of the patch to extract from the image.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.

  Returns:
    images: A `Tensor` of size [batch_size, patch_size, patch_size, 3] and
      type tf.float32. Output pixel values are in (-1, 1).
    labels: A `Tensor` of size [batch_size] with type tf.float32.

  Raises:
    ValueError: If `split` is not 'train' or 'validation'.
  """
  ds = provide_dataset(split, batch_size, patch_size, num_parallel_calls,
                       shuffle)

  next_batch = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
  images, labels = next_batch['images'], next_batch['labels']

  return images, labels


def float_image_to_uint8(image):
  """Convert float image in ~[-0.9, 0.9) to [0, 255] uint8.

  Args:
    image: An image tensor. Values should be in [-0.9, 0.9).

  Returns:
    Input image cast to uint8 and with integer values in [0, 255].
  """
  image = (image * 142.0) + 128.0
  return tf.cast(image, tf.uint8)
