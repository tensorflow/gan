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

"""Contains code for loading and preprocessing image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def normalize_image(image):
  """Rescale from range [0, 255] to [-1, 1]."""
  return (tf.cast(image, tf.float32) - 127.5) / 127.5


def undo_normalize_image(normalized_image):
  """Convert to a numpy array that can be read by PIL."""
  # Convert from NHWC to HWC.
  normalized_image = np.squeeze(normalized_image, axis=0)
  return np.uint8(normalized_image * 127.5 + 127.5)


def _sample_patch(image, patch_size):
  """Crop image to square shape and resize it to `patch_size`.

  Args:
    image: A 3D `Tensor` of HWC format.
    patch_size: A Python scalar.  The output image size.

  Returns:
    A 3D `Tensor` of HWC format which has the shape of
    [patch_size, patch_size, 3].
  """
  image_shape = tf.shape(input=image)
  height, width = image_shape[0], image_shape[1]
  target_size = tf.minimum(height, width)
  image = tf.image.resize_with_crop_or_pad(image, target_size, target_size)
  # tf.image.resize_area only accepts 4D tensor, so expand dims first.
  image = tf.expand_dims(image, axis=0)
  image = tf.compat.v1.image.resize(image, [patch_size, patch_size])
  image = tf.squeeze(image, axis=0)
  # Force image num_channels = 3
  image = tf.tile(image, [1, 1, tf.maximum(1, 4 - tf.shape(input=image)[2])])
  image = tf.slice(image, [0, 0, 0], [patch_size, patch_size, 3])
  return image


def full_image_to_patch(image, patch_size):
  image = normalize_image(image)
  # Sample a patch of fixed size.
  image_patch = _sample_patch(image, patch_size)
  image_patch.shape.assert_is_compatible_with([patch_size, patch_size, 3])
  return image_patch


def _provide_custom_dataset(image_file_pattern, num_threads=1):
  """Provides batches of custom image data.

  Args:
    image_file_pattern: A string of glob pattern of image files.
    num_threads: Number of mapping threads.  Defaults to 1.

  Returns:
    A tf.data.Dataset with image elements.
  """
  filenames_ds = tf.data.Dataset.list_files(image_file_pattern)
  bytes_ds = filenames_ds.map(tf.io.read_file, num_parallel_calls=num_threads)
  images_ds = bytes_ds.map(
      tf.image.decode_image, num_parallel_calls=num_threads)
  return images_ds


def _preprocess_datasets(dataset, batch_size, shuffle=True, num_threads=1,
                         patch_size=128):
  """Run prepreocessing on a list of datasets.

  Args:
    dataset: A dataset with a single element.
    batch_size: The number of images in each batch.
    shuffle: Whether to shuffle the read images.  Defaults to True.
    num_threads: Number of mapping threads.  Defaults to 1.
    patch_size: Size of the path to extract from the image.  Defaults to 128.

  Returns:
    A list of processed datasets. Each dataset has a single entry with shape
    [batch_size, batch_size, batch_size, channels].s
  """
  patches_ds = dataset.map(
      lambda img: full_image_to_patch(img, patch_size),
      num_parallel_calls=num_threads)
  patches_ds = patches_ds.repeat()

  if shuffle:
    patches_ds = patches_ds.shuffle(5 * batch_size)

  patches_ds = patches_ds.prefetch(5 * batch_size)
  patches_ds = patches_ds.batch(batch_size)

  return patches_ds


def provide_custom_datasets(batch_size,
                            image_file_patterns=None,
                            shuffle=True,
                            num_threads=1,
                            patch_size=128):
  """Provides multiple batches of custom image data.

  Args:
    batch_size: The number of images in each batch.
    image_file_patterns: A list of glob patterns of image files. If `None`, use
      the 'Horses and Zebras' datasets from `tensorflow_datasets`.
    shuffle: Whether to shuffle the read images.  Defaults to True.
    num_threads: Number of prefetching threads.  Defaults to 1.
    patch_size: Size of the patch to extract from the image.  Defaults to 128.

  Returns:
    A list of tf.data.Datasets the same number as `image_file_patterns`. Each
    of the datasets have `Tensor`'s in the list has a shape of
    [batch_size, patch_size, patch_size, 3] representing a batch of images.

  Raises:
    ValueError: If image_file_patterns is not a list, tuple, or `None`.
  """
  if image_file_patterns and not isinstance(image_file_patterns, (list, tuple)):
    raise ValueError(
        '`image_file_patterns` should be either list or tuple, but was {}.'
        .format(type(image_file_patterns)))
  images_ds = []
  if image_file_patterns:
    for pattern in image_file_patterns:
      images_ds.append(
          _provide_custom_dataset(image_file_pattern=pattern,
                                  num_threads=num_threads))
  else:
    ds_dict = tfds.load('cycle_gan', shuffle_files=shuffle)
    def _img(x):
      return x['image']
    images_ds = [ds_dict['trainA'].map(_img, num_parallel_calls=num_threads),
                 ds_dict['trainB'].map(_img, num_parallel_calls=num_threads)]
  return [_preprocess_datasets(x, batch_size, shuffle, num_threads, patch_size)
          for x in images_ds]


def provide_custom_data(batch_size,
                        image_file_patterns=None,
                        shuffle=True,
                        num_threads=1,
                        patch_size=128):
  """Provides multiple batches of custom image data.

  Args:
    batch_size: The number of images in each batch.
    image_file_patterns: A list of glob patterns of image files. If `None`, use
      the 'Horses and Zebras' datasets from `tensorflow_datasets`.
    shuffle: Whether to shuffle the read images.  Defaults to True.
    num_threads: Number of prefetching threads.  Defaults to 1.
    patch_size: Size of the patch to extract from the image.  Defaults to 128.

  Returns:
    A list of float `Tensor`s with the same size of `image_file_patterns`. Each
    of the `Tensor` in the list has a shape of
    [batch_size, patch_size, patch_size, 3] representing a batch of images. As a
    side effect, the tf.Dataset initializer is added to the
    tf.GraphKeys.TABLE_INITIALIZERS collection.

  Raises:
    ValueError: If image_file_patterns is not a list or tuple.
  """
  datasets = provide_custom_datasets(batch_size, image_file_patterns, shuffle,
                                     num_threads, patch_size)

  tensors = []
  for ds in datasets:
    iterator = tf.compat.v1.data.make_initializable_iterator(ds)
    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS,
                                   iterator.initializer)
    tensors.append(iterator.get_next())

  # Add batch size to shape information.
  for ts in tensors:
    if ts.shape:  # If we know the number of dimensions.
      partial_shape = ts.shape.as_list()
      partial_shape[0] = batch_size
      ts.set_shape(partial_shape)

  return tensors
