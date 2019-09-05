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

"""Trains a CycleGAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.cyclegan import data_provider
from tensorflow_gan.examples.cyclegan import networks

HParams = collections.namedtuple('HParams', [
    'image_set_x_file_pattern',
    'image_set_y_file_pattern',
    'batch_size',
    'patch_size',
    'master',
    'train_log_dir',
    'generator_lr',
    'discriminator_lr',
    'max_number_of_steps',
    'ps_replicas',
    'task',
    'cycle_consistency_loss_weight',
])


def _get_data(image_set_x_file_pattern, image_set_y_file_pattern, batch_size,
              patch_size):
  """Returns image TEnsors from a custom provider or TFDS."""
  if image_set_x_file_pattern and image_set_y_file_pattern:
    image_file_patterns = [image_set_x_file_pattern, image_set_y_file_pattern]
  else:
    if image_set_x_file_pattern or image_set_y_file_pattern:
      raise ValueError('Both image patterns or neither must be provided.')
    image_file_patterns = None
  images_x, images_y = data_provider.provide_custom_data(
      batch_size=batch_size,
      image_file_patterns=image_file_patterns,
      patch_size=patch_size)

  return images_x, images_y


def _define_model(images_x, images_y):
  """Defines a CycleGAN model that maps between images_x and images_y.

  Args:
    images_x: A 4D float `Tensor` of NHWC format.  Images in set X.
    images_y: A 4D float `Tensor` of NHWC format.  Images in set Y.

  Returns:
    A `CycleGANModel` namedtuple.
  """
  cyclegan_model = tfgan.cyclegan_model(
      generator_fn=networks.generator,
      discriminator_fn=networks.discriminator,
      data_x=images_x,
      data_y=images_y)

  # Add summaries for generated images.
  tfgan.eval.add_cyclegan_image_summaries(cyclegan_model)

  return cyclegan_model


def _get_lr(base_lr, max_number_of_steps):
  """Returns a learning rate `Tensor`.

  Args:
    base_lr: A scalar float `Tensor` or a Python number.  The base learning
      rate.
    max_number_of_steps: The maximum number of steps to train.

  Returns:
    A scalar float `Tensor` of learning rate which equals `base_lr` when the
    global training step is less than max_number_of_steps / 2,
    afterwards it linearly decays to zero.
  """
  global_step = tf.compat.v1.train.get_or_create_global_step()
  lr_constant_steps = max_number_of_steps // 2

  def _lr_decay():
    return tf.compat.v1.train.polynomial_decay(
        learning_rate=base_lr,
        global_step=(global_step - lr_constant_steps),
        decay_steps=(max_number_of_steps - lr_constant_steps),
        end_learning_rate=0.0)

  return tf.cond(
      pred=global_step < lr_constant_steps,
      true_fn=lambda: base_lr,
      false_fn=_lr_decay)


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
  # beta1 follows
  # https://github.com/junyanz/CycleGAN/blob/master/options.lua
  gen_opt = tf.compat.v1.train.AdamOptimizer(
      gen_lr, beta1=0.5, use_locking=True)
  dis_opt = tf.compat.v1.train.AdamOptimizer(
      dis_lr, beta1=0.5, use_locking=True)
  return gen_opt, dis_opt


def _define_train_ops(cyclegan_model, cyclegan_loss, hparams):
  """Defines train ops that trains `cyclegan_model` with `cyclegan_loss`.

  Args:
    cyclegan_model: A `CycleGANModel` namedtuple.
    cyclegan_loss: A `CycleGANLoss` namedtuple containing all losses for
      `cyclegan_model`.
    hparams: An HParams instance containing the hyperparameters for training.

  Returns:
    A `GANTrainOps` namedtuple.
  """
  gen_lr = _get_lr(hparams.generator_lr, hparams.max_number_of_steps)
  dis_lr = _get_lr(hparams.discriminator_lr, hparams.max_number_of_steps)
  gen_opt, dis_opt = _get_optimizer(gen_lr, dis_lr)
  train_ops = tfgan.gan_train_ops(
      cyclegan_model,
      cyclegan_loss,
      generator_optimizer=gen_opt,
      discriminator_optimizer=dis_opt,
      summarize_gradients=True,
      colocate_gradients_with_ops=True,
      aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

  tf.compat.v1.summary.scalar('generator_lr', gen_lr)
  tf.compat.v1.summary.scalar('discriminator_lr', dis_lr)
  return train_ops


def train(hparams):
  """Trains a CycleGAN.

  Args:
    hparams: An HParams instance containing the hyperparameters for training.
  """
  if not tf.io.gfile.exists(hparams.train_log_dir):
    tf.io.gfile.makedirs(hparams.train_log_dir)

  with tf.device(tf.compat.v1.train.replica_device_setter(hparams.ps_replicas)):
    with tf.compat.v1.name_scope('inputs'), tf.device('/cpu:0'):
      images_x, images_y = _get_data(hparams.image_set_x_file_pattern,
                                     hparams.image_set_y_file_pattern,
                                     hparams.batch_size, hparams.patch_size)

    # Define CycleGAN model.
    cyclegan_model = _define_model(images_x, images_y)

    # Define CycleGAN loss.
    cyclegan_loss = tfgan.cyclegan_loss(
        cyclegan_model,
        cycle_consistency_loss_weight=hparams.cycle_consistency_loss_weight,
        tensor_pool_fn=tfgan.features.tensor_pool)

    # Define CycleGAN train ops.
    train_ops = _define_train_ops(cyclegan_model, cyclegan_loss, hparams)

    # Training
    train_steps = tfgan.GANTrainSteps(1, 1)
    status_message = tf.strings.join([
        'Starting train step: ',
        tf.as_string(tf.compat.v1.train.get_or_create_global_step())
    ],
                                     name='status_message')
    if not hparams.max_number_of_steps:
      return
    tfgan.gan_train(
        train_ops,
        hparams.train_log_dir,
        get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
        hooks=[
            tf.estimator.StopAtStepHook(num_steps=hparams.max_number_of_steps),
            tf.estimator.LoggingTensorHook({'status_message': status_message},
                                           every_n_iter=10)
        ],
        master=hparams.master,
        is_chief=hparams.task == 0)
