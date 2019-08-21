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

"""StarGAN data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_gan.examples.cyclegan import data_provider


def provide_dataset(split, batch_size, patch_size, num_parallel_calls=None,
                    shuffle=True,
                    domains=('Black_Hair', 'Blond_Hair', 'Brown_Hair')):
  """Provides batches of CelebA image patches.

  Args:
    split: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    patch_size: Python int. The patch size to extract.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.
    domains: Name of domains to transform between. Must be in Celeb A dataset.

  Returns:
    A tf.data.Dataset with:
      * images:  `Tensor` of size [batch_size, 32, 32, 3] and type tf.float32.
          Output pixel values are in [-1, 1].
      * labels: A `Tensor` of size [batch_size, 10] of one-hot label
          encodings with type tf.int32, or a `Tensor` of size [batch_size],
          depending on the value of `one_hot`.

  Raises:
    ValueError: If `split` isn't `train` or `test`.
  """
  ds = tfds.load('celeb_a', split=split, shuffle_files=shuffle)

  def _filter_pred(attribute):
    def _filter(element):
      return element['attributes'][attribute]
    return _filter
  dss = tuple([ds.filter(_filter_pred(attribute)) for attribute in domains])
  ds = tf.data.Dataset.zip(dss)

  def _preprocess(*elements):
    """Map elements to the example dicts expected by the model."""
    output_dict = {}
    num_domains = len(elements)
    for idx, (domain, elem) in enumerate(zip(domains, elements)):
      uint8_img = elem['image']
      patch = data_provider.full_image_to_patch(uint8_img, patch_size)
      label = tf.one_hot(idx, num_domains)
      output_dict[domain] = {'images': patch, 'labels': label}
    return output_dict

  ds = (ds
        .map(_preprocess, num_parallel_calls=num_parallel_calls)
        .cache()
        .repeat())
  if shuffle:
    ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
  ds = (ds
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

  return ds


def provide_data(split, batch_size, patch_size, num_parallel_calls=None,
                 shuffle=True,
                 domains=('Black_Hair', 'Blond_Hair', 'Brown_Hair')):
  """Provides batches of CelebA image patches.

  Args:
    split: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    patch_size: Python int. The patch size to extract.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.
    domains: Name of domains to transform between. Must be in Celeb A dataset.

  Returns:
    A tf.data.Dataset with:
      * images:  `Tensor` of size [batch_size, patch_size, patch_size, 3] and
          type tf.float32. Output pixel values are in [-1, 1].
      * labels: A `Tensor` of size [batch_size, 10] of one-hot label
          encodings with type tf.int32, or a `Tensor` of size [batch_size],
          depending on the value of `one_hot`.

  Raises:
    ValueError: If `split` isn't `train` or `test`.
  """
  ds = provide_dataset(split, batch_size, patch_size, num_parallel_calls,
                       shuffle, domains)

  next_batch = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
  domains = next_batch.keys()
  images = [next_batch[domain]['images'] for domain in domains]
  labels = [next_batch[domain]['labels'] for domain in domains]

  return images, labels
