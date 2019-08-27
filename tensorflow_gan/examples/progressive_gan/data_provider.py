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

# python2 python3
"""Loading and preprocessing image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _to_int32(tensor):
  return tf.cast(tensor, tf.int32)


def normalize_image(image):
  """Rescales image from range [0, 255] to [-1, 1]."""
  return (tf.cast(image, tf.float32) - 127.5) / 127.5


def sample_patch(image, patch_height, patch_width, colors):
  """Crops image to the desired aspect ratio shape and resizes it.

  If the image has shape H x W, crops a square in the center of
  shape min(H,W) x min(H,W).

  Args:
    image: A 3D `Tensor` of HWC format.
    patch_height: A Python integer. The output images height.
    patch_width: A Python integer. The output images width.
    colors: Number of output image channels.

  Returns:
    A 3D `Tensor` of HWC format with shape [patch_height, patch_width, colors].
  """
  image_shape = tf.shape(input=image)
  h, w = image_shape[0], image_shape[1]

  h_major_target_h = h
  h_major_target_w = tf.maximum(1, _to_int32((h * patch_width) / patch_height))
  w_major_target_h = tf.maximum(1, _to_int32((w * patch_height) / patch_width))
  w_major_target_w = w
  target_hw = tf.cond(
      pred=h_major_target_w <= w,
      true_fn=lambda: tf.convert_to_tensor(  # pylint:disable=g-long-lambda
          value=[h_major_target_h, h_major_target_w]),
      false_fn=lambda: tf.convert_to_tensor(  # pylint:disable=g-long-lambda
          value=[w_major_target_h, w_major_target_w]))
  # Cut a patch of shape (target_h, target_w).
  image = tf.image.resize_with_crop_or_pad(image, target_hw[0], target_hw[1])
  # Resize the cropped image to (patch_h, patch_w).
  image = tf.compat.v1.image.resize([image], [patch_height, patch_width])[0]
  # Force number of channels: repeat the channel dimension enough
  # number of times and then slice the first `colors` channels.
  num_repeats = _to_int32(tf.math.ceil(colors / image_shape[2]))
  image = tf.tile(image, [1, 1, num_repeats])
  image = tf.slice(image, [0, 0, 0], [-1, -1, colors])
  image.set_shape([patch_height, patch_width, colors])
  return image


def _standard_ds_pipeline(ds, batch_size, patch_height, patch_width, colors,
                          num_parallel_calls, shuffle):
  """Efficiently process and batch a tf.data.Dataset."""

  def _preprocess(element):
    """Map elements to the example dicts expected by the model."""
    images = normalize_image(element['image'])
    images = sample_patch(images, patch_height, patch_width, colors)
    return {'images': images}

  ds = (
      ds.map(_preprocess,
             num_parallel_calls=num_parallel_calls).cache().repeat())
  if shuffle:
    ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
  ds = (
      ds.batch(batch_size,
               drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
  return ds


def provide_dataset(split,
                    batch_size,
                    patch_height=32,
                    patch_width=32,
                    colors=3,
                    num_parallel_calls=None,
                    shuffle=True):
  """Provides batches of images.

  Args:
    split: Either 'train' or 'test'.
    batch_size: A Python integer. The number of images in each batch.
    patch_height: A Python integer. The read images height. Defaults to 32.
    patch_width: A Python integer. The read images width. Defaults to 32.
    colors: Number of channels. Defaults to 3.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.

  Returns:
    A tf.data.Dataset with:
      * images: A `Tensor` of size [batch_size, 32, 32, 3] and type tf.float32.
          Output pixel values are in [-1, 1].

  Raises:
    ValueError: If `dataset_name` is invalid.
    ValueError: If `split_name` is invalid.
  """
  ds = tfds.load('cifar10', split=split, shuffle_files=shuffle)
  ds = _standard_ds_pipeline(ds, batch_size, patch_height, patch_width, colors,
                             num_parallel_calls, shuffle)

  return ds


def provide_data(split,
                 batch_size,
                 patch_height=32,
                 patch_width=32,
                 colors=3,
                 num_parallel_calls=None,
                 shuffle=True):
  """Provides batches of CIFAR10 digits.

  Args:
    split: Either 'train' or 'test'.
    batch_size: A Python integer. The number of images in each batch.
    patch_height: A Python integer. The read images height. Defaults to 32.
    patch_width: A Python integer. The read images width. Defaults to 32.
    colors: Number of channels. Defaults to 3.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.

  Returns:
    images: A `Tensor` of size [batch_size, 32, 32, 3] and type tf.float32.
      Output pixel values are in [-1, 1].

  Raises:
    ValueError: If `dataset_name` is invalid.
    ValueError: If `split_name` is invalid.
  """
  ds = provide_dataset(split, batch_size, patch_height, patch_width, colors,
                       num_parallel_calls, shuffle)

  next_batch = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
  images = next_batch['images']

  return images


def provide_data_from_image_files(file_pattern,
                                  batch_size=32,
                                  patch_height=32,
                                  patch_width=32,
                                  colors=3,
                                  num_parallel_calls=None,
                                  shuffle=True):
  """Provides a batch of image data from image files.

  Args:
    file_pattern: A file pattern (glob).
    batch_size: The number of images in each minibatch.  Defaults to 32.
    patch_height: A Python integer. The read images height. Defaults to 32.
    patch_width: A Python integer. The read images width. Defaults to 32.
    colors: Number of channels. Defaults to 3.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.

  Returns:
    A float `Tensor` of shape [batch_size, patch_height, patch_width, 3]
    representing a batch of images.
  """
  dataset_files = tf.io.gfile.glob(file_pattern)
  if shuffle:
    dataset_files = np.random.permutation(dataset_files)
  np_data = np.array([matplotlib.pyplot.imread(x) for x in dataset_files])
  ds = tf.data.Dataset.from_tensor_slices(np_data)
  if shuffle:
    ds = ds.shuffle(reshuffle_each_iteration=True)

  def _make_dict(img):
    if img.shape.ndims == 2:
      img = tf.expand_dims(img, -1)
    return {'image': img}

  ds = ds.map(_make_dict, num_parallel_calls=num_parallel_calls)
  ds = _standard_ds_pipeline(ds, batch_size, patch_height, patch_width, colors,
                             num_parallel_calls, shuffle)

  next_batch = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
  images = next_batch['images']

  return images
