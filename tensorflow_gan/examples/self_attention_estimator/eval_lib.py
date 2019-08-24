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
import tensorflow as tf
from tensorflow_gan.examples.self_attention_estimator import data_provider
import tensorflow_gan as tfgan  # tf


# TODO(joelshor, marvinritter): Make a combined  TPU/CPU/GPU graph the TF-GAN
# default, so this isn't necessary.
def default_graph_def_fn():
  url = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05_v4.tar.gz'
  graph_def = 'inceptionv1_for_inception_score_tpu.pb'
  return tfgan.eval.get_graph_def_from_url_tarball(
      url, graph_def, os.path.basename(url))




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
  def sample_fn(_):
    images = get_images_fn()
    inception_img_sz = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
    larger_images = tf.compat.v1.image.resize(
        images, [inception_img_sz, inception_img_sz],
        method=tf.image.ResizeMethod.BILINEAR)
    return larger_images

  if get_logits:
    output_tensor = (tfgan.eval.INCEPTION_OUTPUT,
                     tfgan.eval.INCEPTION_FINAL_POOL)
  else:
    output_tensor = tfgan.eval.INCEPTION_FINAL_POOL
  output = tfgan.eval.sample_and_run_inception(
      sample_fn,
      sample_inputs=[1.0] * num_batches,  # dummy inputs
      output_tensor=output_tensor,
      default_graph_def_fn=default_graph_def_fn)

  return output


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
