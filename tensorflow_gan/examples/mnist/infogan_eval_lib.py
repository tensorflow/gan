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

"""Evaluates an InfoGAN TFGAN trained MNIST model.

The image visualizations, as in https://arxiv.org/abs/1606.03657, show the
effect of varying a specific latent variable on the image. Each visualization
focuses on one of the three structured variables. Columns have two of the three
variables fixed, while the third one is varied. Different rows have different
random samples from the remaining latents.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples import evaluation_helper as evaluation
from tensorflow_gan.examples.mnist import data_provider
from tensorflow_gan.examples.mnist import networks
from tensorflow_gan.examples.mnist import util

HParams = collections.namedtuple('HParams', [
    'checkpoint_dir', 'eval_dir', 'noise_samples', 'unstructured_noise_dims',
    'continuous_noise_dims', 'max_number_of_evaluations',
    'write_to_disk'
])

CAT_SAMPLE_POINTS = np.arange(0, 10)
CONT_SAMPLE_POINTS = np.linspace(-2.0, 2.0, 10)


def evaluate(hparams, run_eval_loop=True):
  """Runs an evaluation loop.

  Args:
    hparams: An HParams instance containing the eval hyperparameters.
    run_eval_loop: Whether to run the full eval loop. Set to False for testing.
  """
  with tf.compat.v1.name_scope('inputs'):
    noise_args = (hparams.noise_samples, CAT_SAMPLE_POINTS, CONT_SAMPLE_POINTS,
                  hparams.unstructured_noise_dims,
                  hparams.continuous_noise_dims)
    # Use fixed noise vectors to illustrate the effect of each dimension.
    display_noise1 = util.get_eval_noise_categorical(*noise_args)
    display_noise2 = util.get_eval_noise_continuous_dim1(*noise_args)
    display_noise3 = util.get_eval_noise_continuous_dim2(*noise_args)
    _validate_noises([display_noise1, display_noise2, display_noise3])

  # Visualize the effect of each structured noise dimension on the generated
  # image.
  def generator_fn(inputs):
    return networks.infogan_generator(
        inputs, len(CAT_SAMPLE_POINTS), is_training=False)

  with tf.compat.v1.variable_scope(
      'Generator') as genscope:  # Same scope as in training.
    categorical_images = generator_fn(display_noise1)
  reshaped_categorical_img = tfgan.eval.image_reshaper(
      categorical_images, num_cols=len(CAT_SAMPLE_POINTS))
  tf.compat.v1.summary.image(
      'categorical', reshaped_categorical_img, max_outputs=1)

  with tf.compat.v1.variable_scope(genscope, reuse=True):
    continuous1_images = generator_fn(display_noise2)
  reshaped_continuous1_img = tfgan.eval.image_reshaper(
      continuous1_images, num_cols=len(CONT_SAMPLE_POINTS))
  tf.compat.v1.summary.image(
      'continuous1', reshaped_continuous1_img, max_outputs=1)

  with tf.compat.v1.variable_scope(genscope, reuse=True):
    continuous2_images = generator_fn(display_noise3)
  reshaped_continuous2_img = tfgan.eval.image_reshaper(
      continuous2_images, num_cols=len(CONT_SAMPLE_POINTS))
  tf.compat.v1.summary.image(
      'continuous2', reshaped_continuous2_img, max_outputs=1)

  # Evaluate image quality.
  all_images = tf.concat(
      [categorical_images, continuous1_images, continuous2_images], 0)
  tf.compat.v1.summary.scalar(
      'MNIST_Classifier_score', util.mnist_score(all_images))

  # Write images to disk.
  image_write_ops = []
  if hparams.write_to_disk:
    image_write_ops.append(
        _get_write_image_ops(hparams.eval_dir, 'categorical_infogan.png',
                             reshaped_categorical_img[0]))
    image_write_ops.append(
        _get_write_image_ops(hparams.eval_dir, 'continuous1_infogan.png',
                             reshaped_continuous1_img[0]))
    image_write_ops.append(
        _get_write_image_ops(hparams.eval_dir, 'continuous2_infogan.png',
                             reshaped_continuous2_img[0]))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop:
    return
  evaluation.evaluate_repeatedly(
      hparams.checkpoint_dir,
      hooks=[
          evaluation.SummaryAtEndHook(hparams.eval_dir),
          evaluation.StopAfterNEvalsHook(1)
      ],
      eval_ops=image_write_ops,
      max_number_of_evaluations=hparams.max_number_of_evaluations)


def _validate_noises(noises):
  """Sanity check on constructed noise tensors.

  Args:
    noises: List of 3-tuples of noise vectors.
  """
  assert isinstance(noises, (list, tuple))
  for noise_l in noises:
    assert len(noise_l) == 3
    assert isinstance(noise_l[0], np.ndarray)
    batch_dim = noise_l[0].shape[0]
    for i, noise in enumerate(noise_l):
      assert isinstance(noise, np.ndarray)
      # Check that batch dimensions are all the same.
      assert noise.shape[0] == batch_dim

      # Check that shapes for corresponding noises are the same.
      assert noise.shape == noises[0][i].shape


def _get_write_image_ops(eval_dir, filename, images):
  """Create Ops that write images to disk."""
  return tf.io.write_file(
      '%s/%s' % (eval_dir, filename),
      tf.image.encode_png(data_provider.float_image_to_uint8(images)))
