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

"""Evaluates a TF-GAN trained MNIST model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples import evaluation_helper as evaluation
from tensorflow_gan.examples.mnist import data_provider
from tensorflow_gan.examples.mnist import networks
from tensorflow_gan.examples.mnist import util

HParams = collections.namedtuple('HParams', [
    'checkpoint_dir', 'eval_dir', 'dataset_dir', 'num_images_generated',
    'eval_real_images', 'noise_dims', 'max_number_of_evaluations',
    'write_to_disk'
])


def evaluate(hparams, run_eval_loop=True):
  """Runs an evaluation loop.

  Args:
    hparams: An HParams instance containing the eval hyperparameters.
    run_eval_loop: Whether to run the full eval loop. Set to False for testing.
  """
  # Fetch real images.
  with tf.compat.v1.name_scope('inputs'):
    real_images, _ = data_provider.provide_data('train',
                                                hparams.num_images_generated,
                                                hparams.dataset_dir)

  image_write_ops = None
  if hparams.eval_real_images:
    tf.compat.v1.summary.scalar(
        'MNIST_Classifier_score', util.mnist_score(real_images))
  else:
    # In order for variables to load, use the same variable scope as in the
    # train job.
    with tf.compat.v1.variable_scope('Generator'):
      images = networks.unconditional_generator(
          tf.random.normal([hparams.num_images_generated, hparams.noise_dims]),
          is_training=False)
    tf.compat.v1.summary.scalar(
        'MNIST_Frechet_distance',
        util.mnist_frechet_distance(real_images, images))
    tf.compat.v1.summary.scalar(
        'MNIST_Classifier_score', util.mnist_score(images))
    if hparams.num_images_generated >= 100 and hparams.write_to_disk:
      reshaped_images = tfgan.eval.image_reshaper(
          images[:100, ...], num_cols=10)
      uint8_images = data_provider.float_image_to_uint8(reshaped_images)
      image_write_ops = tf.io.write_file(
          '%s/%s' % (hparams.eval_dir, 'unconditional_gan.png'),
          tf.image.encode_png(uint8_images[0]))

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
