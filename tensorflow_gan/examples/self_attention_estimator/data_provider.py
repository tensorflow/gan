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

from absl import flags
import tensorflow as tf  # tf
import tensorflow_datasets as tfds

from tensorflow_gan.examples import compat_utils

IMG_SIZE = 128

flags.DEFINE_string('imagenet_data_dir', None,
                    'A directory for TFDS ImageNet. If `None`, use default.')


def provide_dataset(batch_size, shuffle_buffer_size, split='train'):
  """Provides dataset of ImageNet digits that were preprocessed by the Red Team.

  Args:
    batch_size: The number of images in each batch.
    shuffle_buffer_size: The number of records to load before shuffling. Larger
      means more likely randomization.
    split: A tfds split. If 'train', dataset is shuffled. Otherwise, it's
      deterministic.
  Returns:
    A dataset of num_batches batches of size batch_size of images and labels.
  """
  shuffle = (split in ['train', tfds.Split.TRAIN])
  dataset = _load_imagenet_dataset(split, flags.FLAGS.imagenet_data_dir,
                                   shuffle_files=shuffle)
  if shuffle:
    dataset = dataset.apply(
        tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size))
  else:
    dataset = dataset.repeat()
  dataset = (dataset.map(_preprocess_dataset_record_fn(IMG_SIZE),
                         num_parallel_calls=16 if shuffle else None)
             .batch(batch_size, drop_remainder=True))
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def provide_data(batch_size,
                 num_batches,
                 shuffle_buffer_size,
                 split='train'):
  """Provides batches of ImageNet digits that were preprocessed by the Red Team.

  Args:
    batch_size: The number of images in each batch.
    num_batches: The number of batches to return.
    shuffle_buffer_size: The number of records to load before shuffling. Larger
      means more likely randomization.
    split: A tfds split.

  Returns:
    A list of num_batches batches of size batch_size. Each element in the
    returned list is a tuple of a batch of images and a batch of the respective
    labels.
  """
  dataset = provide_dataset(batch_size * num_batches, shuffle_buffer_size,
                            split)
  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
  images, labels = iterator.get_next()
  images = tf.reshape(
      images, shape=[num_batches, batch_size, IMG_SIZE, IMG_SIZE, 3])
  labels = tf.reshape(labels, shape=[num_batches, batch_size, 1])
  batches = list(
      zip(
          tf.unstack(images, num=num_batches),
          tf.unstack(labels, num=num_batches)))
  return batches


def _load_imagenet_dataset(split, data_dir=None, shuffle_files=False):
  return tfds.load('imagenet2012', split=split, data_dir=data_dir,
                   shuffle_files=shuffle_files)


def _preprocess_dataset_record_fn(image_size):
  """Returns function for processing the elements of the imagenet dataset."""

  def _process_record(record):
    """Takes the largest central square and resamples to image_size."""
    # Based on
    # https://github.com/openai/improved-gan/blob/master/imagenet/convert_imagenet_to_records.py
    image = record['image']
    image_shape = tf.cast(tf.shape(input=image), tf.float32)
    box_size = tf.math.minimum(image_shape[0], image_shape[1])
    # Since we assume the box is centered we have:
    # 2 * box_x_min + box_size == box_width,
    # 2 * box_y_min + box_size == box_height.
    # tf.math.ceil is used for consistency with the improved-gan implementation.
    box_y_min = tf.math.ceil(0.5 * (image_shape[0] - box_size))
    box_x_min = tf.math.ceil(0.5 * (image_shape[1] - box_size))
    box_y_max = box_y_min + box_size - 1
    box_x_max = box_x_min + box_size - 1
    # Normalize with the inverse of the trasform done by crop_and_resize.
    normalized_y_min = box_y_min / (image_shape[0] - 1)
    normalized_x_min = box_x_min / (image_shape[1] - 1)
    normalized_y_max = box_y_max / (image_shape[0] - 1)
    normalized_x_max = box_x_max / (image_shape[1] - 1)
    image = compat_utils.crop_and_resize([image],
                                         boxes=[[
                                             normalized_y_min, normalized_x_min,
                                             normalized_y_max, normalized_x_max
                                         ]],
                                         box_ind=[0],
                                         crop_size=[image_size, image_size])
    # crop_and_resize returns a tensor of type tf.float32.
    image = tf.squeeze(image, axis=0)
    image = image * (2. / 255) - 1.
    label = tf.cast(record['label'], tf.int32)
    return image, label

  return _process_record
