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

"""Trains a StarGAN model using tfgan.estimator.StarGANEstimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
import PIL
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.stargan import network
from tensorflow_gan.examples.stargan_estimator import data_provider

# FLAGS for data.
flags.DEFINE_integer('batch_size_stargan_estimator', 6, 'The number of images in each batch.')
flags.DEFINE_integer('patch_size_stargan_estimator', 128, 'The patch size of images.')

# Write-to-disk flags.
flags.DEFINE_string('output_dir_stargan_estimator', '/tmp/tfgan_logdir/stargan_estimator/out/',
                    'Directory where to write summary image.')

# FLAGS for training hyper-parameters.
flags.DEFINE_float('generator_lr_stargan_estimator', 1e-4, 'The generator learning rate.')
flags.DEFINE_float('discriminator_lr_stargan_estimator', 1e-4, 'The discriminator learning rate.')
flags.DEFINE_integer('max_number_of_steps_stargan_estimator', 1000000,
                     'The maximum number of gradient steps.')
flags.DEFINE_integer('steps_per_eval', 1000,
                     'The number of steps after which we write eval to disk.')
flags.DEFINE_float('adam_beta1_stargan_estimator', 0.5, 'Adam Beta 1 for the Adam optimizer.')
flags.DEFINE_float('adam_beta2_stargan_estimator', 0.999, 'Adam Beta 2 for the Adam optimizer.')
flags.DEFINE_float('gen_disc_step_ratio_stargan_estimator', 0.2,
                   'Generator:Discriminator training step ratio.')

# FLAGS for distributed training.
flags.DEFINE_string('master_stargan_estimator', '', 'Name of the TensorFlow master_stargan_estimator to use.')
flags.DEFINE_integer(
    'ps_task_stargan_estimators', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
flags.DEFINE_integer(
    'task_stargan_estimator', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

FLAGS = flags.FLAGS


def _get_optimizer(gen_lr, dis_lr):
  """Returns generator optimizer and discriminator optimizer.

  Args:
    gen_lr: A scalar float `Tensor` or a Python number.  The Generator learning
      rate.
    dis_lr: A scalar float `Tensor` or a Python number.  The Discriminator
      learning rate.

  Returns:
    A tuple of generator optimizer and discriminator optimizer.
  """
  gen_opt = tf.compat.v1.train.AdamOptimizer(
      gen_lr, beta1=FLAGS.adam_beta1_stargan_estimator, beta2=FLAGS.adam_beta2_stargan_estimator, use_locking=True)
  dis_opt = tf.compat.v1.train.AdamOptimizer(
      dis_lr, beta1=FLAGS.adam_beta1_stargan_estimator, beta2=FLAGS.adam_beta2_stargan_estimator, use_locking=True)
  return gen_opt, dis_opt


def _define_train_step():
  """Get the training step for generator and discriminator for each GAN step.

  Returns:
    GANTrainSteps namedtuple representing the training step configuration.
  """

  if FLAGS.gen_disc_step_ratio_stargan_estimator <= 1:
    discriminator_step = int(1 / FLAGS.gen_disc_step_ratio_stargan_estimator)
    return tfgan.GANTrainSteps(1, discriminator_step)
  else:
    generator_step = int(FLAGS.gen_disc_step_ratio_stargan_estimator)
    return tfgan.GANTrainSteps(generator_step, 1)


def _get_summary_image(estimator, test_images_np):
  """Returns a numpy image of the generate on the test images."""
  num_domains = len(test_images_np)

  img_rows = []
  for img_np in test_images_np:

    def test_input_fn():
      dataset_imgs = [img_np] * num_domains  # pylint:disable=cell-var-from-loop
      dataset_lbls = [tf.one_hot([d], num_domains) for d in xrange(num_domains)]

      # Make into a dataset.
      dataset_imgs = np.stack(dataset_imgs)
      dataset_imgs = np.expand_dims(dataset_imgs, 1)
      dataset_lbls = tf.stack(dataset_lbls)
      unused_tensor = tf.zeros(num_domains)
      return tf.data.Dataset.from_tensor_slices(((dataset_imgs, dataset_lbls),
                                                 unused_tensor))

    prediction_iterable = estimator.predict(test_input_fn)
    predictions = [next(prediction_iterable) for _ in xrange(num_domains)]
    transform_row = np.concatenate([img_np] + predictions, 1)
    img_rows.append(transform_row)

  all_rows = np.concatenate(img_rows, 0)
  # Normalize` [-1, 1] to [0, 1].
  normalized_summary = (all_rows + 1.0) / 2.0
  return normalized_summary


def main(_, override_generator_fn=None, override_discriminator_fn=None):
  # Create directories if not exist.
  if not tf.io.gfile.exists(FLAGS.output_dir_stargan_estimator):
    tf.io.gfile.makedirs(FLAGS.output_dir_stargan_estimator)

  # Make sure steps integers are consistent.
  if FLAGS.max_number_of_steps_stargan_estimator % FLAGS.steps_per_eval != 0:
    raise ValueError('`max_number_of_steps_stargan_estimator` must be divisible by '
                     '`steps_per_eval`.')

  # Create optimizers.
  gen_opt, dis_opt = _get_optimizer(FLAGS.generator_lr_stargan_estimator, FLAGS.discriminator_lr_stargan_estimator)

  # Create estimator.
  stargan_estimator = tfgan.estimator.StarGANEstimator(
      generator_fn=override_generator_fn or network.generator,
      discriminator_fn=override_discriminator_fn or network.discriminator,
      loss_fn=tfgan.stargan_loss,
      generator_optimizer=gen_opt,
      discriminator_optimizer=dis_opt,
      get_hooks_fn=tfgan.get_sequential_train_hooks(_define_train_step()),
      add_summaries=tfgan.estimator.SummaryType.IMAGES)

  # Get input function for training and test images.
  train_input_fn = lambda: data_provider.provide_data(  # pylint:disable=g-long-lambda
      'train', FLAGS.batch_size_stargan_estimator, FLAGS.patch_size_stargan_estimator)
  test_images_np = data_provider.provide_celeba_test_set(FLAGS.patch_size_stargan_estimator)
  filename_str = os.path.join(FLAGS.output_dir_stargan_estimator, 'summary_image_%i.png')

  # Periodically train and write prediction output to disk.
  cur_step = 0
  while cur_step < FLAGS.max_number_of_steps_stargan_estimator:
    cur_step += FLAGS.steps_per_eval
    stargan_estimator.train(train_input_fn, steps=cur_step)
    summary_img = _get_summary_image(stargan_estimator, test_images_np)
    with tf.io.gfile.GFile(filename_str % cur_step, 'w') as f:
      PIL.Image.fromarray((255 * summary_img).astype(np.uint8)).save(f, 'PNG')


if __name__ == '__main__':
  tf.compat.v1.app.run()
