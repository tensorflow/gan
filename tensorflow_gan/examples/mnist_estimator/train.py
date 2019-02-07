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

"""Trains a GANEstimator on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import numpy as np
import scipy.misc
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.mnist import data_provider
from tensorflow_gan.examples.mnist import networks

flags.DEFINE_integer('batch_size', 32,
                     'The number of images in each train batch.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'noise_dims', 64, 'Dimensions of the generator noise vector')

flags.DEFINE_string('output_dir', '/tmp/mnist-estimator/',
                    'Directory where the results are saved to.')

FLAGS = flags.FLAGS


def _get_train_input_fn(batch_size, noise_dims, num_parallel_calls=4):
  def train_input_fn():
    images, _ = data_provider.provide_data(
        'train', batch_size, num_parallel_calls=num_parallel_calls)
    noise = tf.random_normal([batch_size, noise_dims])
    return noise, images
  return train_input_fn


def _get_predict_input_fn(batch_size, noise_dims):
  def predict_input_fn():
    noise = tf.random_normal([batch_size, noise_dims])
    return noise
  return predict_input_fn


def _unconditional_generator(noise, mode):
  """MNIST generator with extra argument for tf.Estimator's `mode`."""
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  return networks.unconditional_generator(noise, is_training=is_training)


def main(_):
  # Initialize GANEstimator with options and hyperparameters.
  gan_estimator = tfgan.estimator.GANEstimator(
      generator_fn=_unconditional_generator,
      discriminator_fn=networks.unconditional_discriminator,
      generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
      discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
      generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
      discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
      add_summaries=tfgan.estimator.SummaryType.IMAGES)

  # Train estimator.
  train_input_fn = _get_train_input_fn(FLAGS.batch_size, FLAGS.noise_dims)
  gan_estimator.train(train_input_fn, max_steps=FLAGS.max_number_of_steps)

  # Run inference.
  predict_input_fn = _get_predict_input_fn(36, FLAGS.noise_dims)
  prediction_iterable = gan_estimator.predict(predict_input_fn)
  predictions = np.array([next(prediction_iterable) for _ in xrange(36)])

  # Nicely tile.
  tiled_image = tfgan.eval.python_image_grid(predictions, grid_shape=(6, 6))

  # Write to disk.
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  scipy.misc.imsave(os.path.join(FLAGS.output_dir, 'unconditional_gan.png'),
                    np.squeeze(tiled_image, axis=2))


if __name__ == '__main__':
  tf.app.run()
