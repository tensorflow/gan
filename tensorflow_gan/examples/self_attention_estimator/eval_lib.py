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

"""Utilities library for evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import PIL

import tensorflow as tf
from tensorflow_gan.examples.self_attention_estimator import data_provider
import tensorflow_gan as tfgan  # tf


def get_activations(get_images_fn, num_batches, get_logits=False):
  """Get Inception activations.

  Use TF-GAN utility to avoid holding images or Inception activations in
  memory all at once.

  Args:
    get_images_fn: A function that takes no arguments and returns images.
    num_batches: The number of batches to fetch at a time.
    get_logits: If `True`, return (logits, pools). Otherwise just return pools.

  Returns:
    1 or 2 Tensors of Inception activations.
  """
  # Image resizing happens inside the Inception SavedModel.
  outputs = tfgan.eval.sample_and_run_inception(
      sample_fn=lambda _: get_images_fn(),
      sample_inputs=[1.0] * num_batches)  # dummy inputs
  if get_logits:
    return outputs['logits'], outputs['pool_3']
  else:
    return outputs['pool_3']


def get_activations_from_dataset(image_ds, num_batches, get_logits=False):
  """Get Inception activations.

  Args:
    image_ds: tf.Dataset for images.
    num_batches: The number of batches to fetch at a time.
    get_logits: If `True`, return (logits, pools). Otherwise just return pools.

  Returns:
    1 or 2 Tensors of Inception activations.
  """
  # TODO(joelshor): Add dataset format checks.
  iterator = tf.compat.v1.data.make_one_shot_iterator(image_ds)

  get_images_fn = iterator.get_next
  return get_activations(get_images_fn, num_batches, get_logits)


def get_real_activations(batch_size,
                         num_batches,
                         shuffle_buffer_size=100000,
                         split='validation',
                         get_logits=False):
  """Fetches batches inception pools and images.

  NOTE: This function runs inference on an Inception network, so it would be
  more efficient to run this on GPU or TPU than on CPU.

  Args:
    batch_size: The number of elements in a single minibatch.
    num_batches: The number of batches to fetch at a time.
    shuffle_buffer_size: The number of records to load before shuffling. Larger
        means more likely randomization.
    split: Shuffle if 'train', else deterministic.
    get_logits: If `True`, return (logits, pools). Otherwise just return pools.

  Returns:
    A Tensor of `real_pools` or (`real_logits`, `real_pools`) with batch
    dimension (batch_size * num_batches).
  """
  ds = data_provider.provide_dataset(batch_size, shuffle_buffer_size, split)
  ds = ds.map(lambda img, lbl: img)  # Remove labels.
  return get_activations_from_dataset(ds, num_batches, get_logits)


def print_debug_statistics(image, labels, dbg_messge_prefix, on_tpu):
  """Adds a Print directive to an image tensor which prints debug statistics."""
  if on_tpu:
    # Print operations are not supported on TPUs.
    return image, labels
  image_means = tf.reduce_mean(input_tensor=image, axis=0, keepdims=True)
  image_vars = tf.reduce_mean(
      input_tensor=tf.math.squared_difference(image, image_means),
      axis=0,
      keepdims=True)
  image = tf.compat.v1.Print(
      image, [
          tf.reduce_mean(input_tensor=image_means),
          tf.reduce_mean(input_tensor=image_vars)
      ],
      dbg_messge_prefix + ' mean and average var',
      first_n=1)
  labels = tf.compat.v1.Print(
      labels, [labels, labels.shape],
      dbg_messge_prefix + ' sparse labels',
      first_n=2)
  return image, labels


def log_and_summarize_variables(var_list, dbg_messge, on_tpu):
  """Logs given variables, summarizes sigma_ratio_vars."""
  tf.compat.v1.logging.info(dbg_messge + str(var_list))
  sigma_ratio_vars = [var for var in var_list if 'sigma_ratio' in var.name]
  tf.compat.v1.logging.info('sigma_ratio_vars %s %s', dbg_messge,
                            sigma_ratio_vars)
  # Reset the name scope so the summary names are displayed as passed to the
  # summary function.
  if not on_tpu:
    # The TPU estimator doesn't support summaries.
    with tf.compat.v1.name_scope(name=None):
      for var in sigma_ratio_vars:
        tf.compat.v1.summary.scalar('sigma_ratio_vars/' + var.name, var)


def predict_and_write_images(estimator, input_fn, model_dir, filename_suffix):
  """Generates images and write them to the model dir.

  Args:
    estimator: An object of type tfgan.estimator.GANEstimator or
      tfgan.estimator.TPUGANEstimator for performing the predictions.
    input_fn: An input_fn function to be used by `estimator.predict`.
    model_dir: The model directory (the images will be saved inside an 'images'
      subdirectory).
    filename_suffix: A suffix to append to the image file names.
  """
  # Generate images.
  image_iterator = estimator.predict(input_fn)
  if isinstance(estimator, tfgan.estimator.TPUGANEstimator):
    predictions = np.array(
        [next(image_iterator)['generated_data'] for _ in range(16)])
  else:
    predictions = np.array([next(image_iterator) for _ in range(16)])
  # Write images to disk.
  output_dir = os.path.join(model_dir, 'images')
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  # Generate a grid of images and write it to disk.
  image_grid = tfgan.eval.python_image_grid(predictions, grid_shape=(4, 4))
  grid_fname = os.path.join(output_dir, 'grid_%s.png' % filename_suffix)
  _write_image_to_disk(image_grid, grid_fname)


def _write_image_to_disk(image, filename):
  with tf.io.gfile.GFile(filename, 'w') as f:
    # Convert tiled_image from float32 in [-1, 1] to unit8 [0, 255].
    img_np = (255 / 2.0) * (image + 1.0)
    pil_image = PIL.Image.fromarray(img_np.astype(np.uint8))
    pil_image.convert('RGB').save(f, 'PNG')
  tf.compat.v1.logging.info('Wrote output to: %s', filename)
