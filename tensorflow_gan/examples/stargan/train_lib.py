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

"""Trains a StarGAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.stargan import data_provider
from tensorflow_gan.examples.stargan import network

HParams = collections.namedtuple('HParams', [
    'batch_size', 'patch_size', 'train_log_dir', 'generator_lr',
    'discriminator_lr', 'max_number_of_steps', 'adam_beta1', 'adam_beta2',
    'gen_disc_step_ratio', 'tf_master', 'ps_replicas', 'task'
])


def _define_model(images, labels):
  """Create the StarGAN Model.

  Args:
    images: `Tensor` or list of `Tensor` of shape (N, H, W, C).
    labels: `Tensor` or list of `Tensor` of shape (N, num_domains).

  Returns:
    `StarGANModel` namedtuple.
  """

  return tfgan.stargan_model(
      generator_fn=network.generator,
      discriminator_fn=network.discriminator,
      input_data=images,
      input_data_domain_label=labels)


def _get_lr(base_lr, max_number_of_steps):
  """Returns a learning rate `Tensor`.

  Args:
    base_lr: A scalar float `Tensor` or a Python number.  The base learning
      rate.
    max_number_of_steps: A Python number. The total number of steps to train.

  Returns:
    A scalar float `Tensor` of learning rate which equals `base_lr` when the
    global training step is less than Fmax_number_of_steps / 2, afterwards
    it linearly decays to zero.
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


def _get_optimizer(gen_lr, dis_lr, beta1, beta2):
  """Returns generator optimizer and discriminator optimizer.

  Args:
    gen_lr: A scalar float `Tensor` or a Python number.  The Generator learning
      rate.
    dis_lr: A scalar float `Tensor` or a Python number.  The Discriminator
      learning rate.
    beta1: A scalar float `Tensor` or a Python number. The beta1 parameter to
      the `AdamOptimizer`.
    beta2: A scalar float `Tensor` or a Python number. The beta2 parameter to
      the `AdamOptimizer`.

  Returns:
    A tuple of generator optimizer and discriminator optimizer.
  """
  gen_opt = tf.compat.v1.train.AdamOptimizer(
      gen_lr, beta1=beta1, beta2=beta2, use_locking=True)
  dis_opt = tf.compat.v1.train.AdamOptimizer(
      dis_lr, beta1=beta1, beta2=beta2, use_locking=True)
  return gen_opt, dis_opt


def _define_train_ops(model, loss, gen_lr, dis_lr, beta1, beta2,
                      max_number_of_steps):
  """Defines train ops that trains `stargan_model` with `stargan_loss`.

  Args:
    model: A `StarGANModel` namedtuple.
    loss: A `StarGANLoss` namedtuple containing all losses for `stargan_model`.
    gen_lr: A scalar float `Tensor` or a Python number.  The Generator base
      learning rate.
    dis_lr: A scalar float `Tensor` or a Python number.  The Discriminator base
      learning rate.
    beta1: A scalar float `Tensor` or a Python number. The beta1 parameter to
      the `AdamOptimizer`.
    beta2: A scalar float `Tensor` or a Python number. The beta2 parameter to
      the `AdamOptimizer`.
    max_number_of_steps: A Python number. The total number of steps to train.

  Returns:
    A `GANTrainOps` namedtuple.
  """

  gen_lr = _get_lr(gen_lr, max_number_of_steps)
  dis_lr = _get_lr(dis_lr, max_number_of_steps)
  gen_opt, dis_opt = _get_optimizer(gen_lr, dis_lr, beta1, beta2)
  train_ops = tfgan.gan_train_ops(
      model,
      loss,
      generator_optimizer=gen_opt,
      discriminator_optimizer=dis_opt,
      summarize_gradients=True,
      colocate_gradients_with_ops=True,
      aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

  tf.compat.v1.summary.scalar('generator_lr', gen_lr)
  tf.compat.v1.summary.scalar('discriminator_lr', dis_lr)

  return train_ops


def _define_train_step(gen_disc_step_ratio):
  """Get the training step for generator and discriminator for each GAN step.

  Args:
    gen_disc_step_ratio: A python number. The ratio of generator to
      discriminator training steps.

  Returns:
    GANTrainSteps namedtuple representing the training step configuration.
  """

  if gen_disc_step_ratio <= 1:
    discriminator_step = int(1 / gen_disc_step_ratio)
    return tfgan.GANTrainSteps(1, discriminator_step)
  else:
    generator_step = int(gen_disc_step_ratio)
    return tfgan.GANTrainSteps(generator_step, 1)


def train(hparams):
  """Trains a StarGAN.

  Args:
    hparams: An HParams instance containing the hyperparameters for training.
  """

  # Create the log_dir if not exist.
  if not tf.io.gfile.exists(hparams.train_log_dir):
    tf.io.gfile.makedirs(hparams.train_log_dir)

  # Shard the model to different parameter servers.
  with tf.device(tf.compat.v1.train.replica_device_setter(hparams.ps_replicas)):

    # Create the input dataset.
    with tf.compat.v1.name_scope('inputs'), tf.device('/cpu:0'):
      images, labels = data_provider.provide_data('train', hparams.batch_size,
                                                  hparams.patch_size)

    # Define the model.
    with tf.compat.v1.name_scope('model'):
      model = _define_model(images, labels)

    # Add image summary.
    tfgan.eval.add_stargan_image_summaries(
        model, num_images=3 * hparams.batch_size, display_diffs=True)

    # Define the model loss.
    loss = tfgan.stargan_loss(model)

    # Define the train ops.
    with tf.compat.v1.name_scope('train_ops'):
      train_ops = _define_train_ops(model, loss, hparams.generator_lr,
                                    hparams.discriminator_lr,
                                    hparams.adam_beta1, hparams.adam_beta2,
                                    hparams.max_number_of_steps)

    # Define the train steps.
    train_steps = _define_train_step(hparams.gen_disc_step_ratio)

    # Define a status message.
    status_message = tf.strings.join([
        'Starting train step: ',
        tf.as_string(tf.compat.v1.train.get_or_create_global_step())
    ],
                                     name='status_message')

    # Train the model.
    tfgan.gan_train(
        train_ops,
        hparams.train_log_dir,
        get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
        hooks=[
            tf.estimator.StopAtStepHook(num_steps=hparams.max_number_of_steps),
            tf.estimator.LoggingTensorHook([status_message], every_n_iter=10)
        ],
        master=hparams.tf_master,
        is_chief=hparams.task == 0)
