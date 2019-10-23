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

"""Evaluates a conditional TF-GAN trained MNIST model."""

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

NUM_CLASSES = 10

HParams = collections.namedtuple('HParams', [
    'checkpoint_dir', 'eval_dir', 'num_images_per_class', 'noise_dims',
    'max_number_of_evaluations', 'write_to_disk'
])


def evaluate(hparams, run_eval_loop=True):
  """Runs an evaluation loop.

  Args:
    hparams: An HParams instance containing the eval hyperparameters.
    run_eval_loop: Whether to run the full eval loop. Set to False for testing.
  """
  with tf.compat.v1.name_scope('inputs'):
    noise, one_hot_labels = _get_generator_inputs(hparams.num_images_per_class,
                                                  NUM_CLASSES,
                                                  hparams.noise_dims)

  # Generate images.
  with tf.compat.v1.variable_scope('Generator'):  # Same scope as in train job.
    images = networks.conditional_generator((noise, one_hot_labels),
                                            is_training=False)

  # Visualize images.
  reshaped_img = tfgan.eval.image_reshaper(
      images, num_cols=hparams.num_images_per_class)
  tf.compat.v1.summary.image('generated_images', reshaped_img, max_outputs=1)

  # Calculate evaluation metrics.
  tf.compat.v1.summary.scalar(
      'MNIST_Classifier_score', util.mnist_score(images))
  tf.compat.v1.summary.scalar(
      'MNIST_Cross_entropy',
      util.mnist_cross_entropy(images, one_hot_labels))

  # Write images to disk.
  image_write_ops = None
  if hparams.write_to_disk:
    image_write_ops = tf.io.write_file(
        '%s/%s' % (hparams.eval_dir, 'conditional_gan.png'),
        tf.image.encode_png(
            data_provider.float_image_to_uint8(reshaped_img[0])))

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


def _get_generator_inputs(num_images_per_class, num_classes, noise_dims):
  """Return generator inputs for evaluation."""
  # Since we want a grid of numbers for the conditional generator, manually
  # construct the desired class labels.
  num_images_generated = num_images_per_class * num_classes
  noise = tf.random.normal([num_images_generated, noise_dims])
  # pylint:disable=g-complex-comprehension
  labels = [
      lbl for lbl in range(num_classes) for _ in range(num_images_per_class)
  ]
  one_hot_labels = tf.one_hot(tf.constant(labels), num_classes)
  return noise, one_hot_labels
