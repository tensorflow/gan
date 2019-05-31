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

"""Trains a generator on CIFAR data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.cifar import data_provider
from tensorflow_gan.examples.cifar import networks

HParams = collections.namedtuple('HParams', [
    'batch_size',
    'max_number_of_steps',
    'generator_lr',
    'discriminator_lr',
    'master',
    'train_log_dir',
    'ps_replicas',
    'task',
])


def train(hparams):
  """Trains a CIFAR10 GAN.

  Args:
    hparams: An HParams instance containing the hyperparameters for training.
  """
  if not tf.io.gfile.exists(hparams.train_log_dir):
    tf.io.gfile.makedirs(hparams.train_log_dir)

  with tf.device(tf.compat.v1.train.replica_device_setter(hparams.ps_replicas)):
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.compat.v1.name_scope('inputs'):
      with tf.device('/cpu:0'):
        images, _ = data_provider.provide_data(
            'train', hparams.batch_size, num_parallel_calls=4)

    # Define the GANModel tuple.
    generator_fn = networks.generator
    discriminator_fn = networks.discriminator
    generator_inputs = tf.random.normal([hparams.batch_size, 64])
    gan_model = tfgan.gan_model(
        generator_fn,
        discriminator_fn,
        real_data=images,
        generator_inputs=generator_inputs)
    tfgan.eval.add_gan_model_image_summaries(gan_model)

    # Get the GANLoss tuple. Use the selected GAN loss functions.
    with tf.compat.v1.name_scope('loss'):
      gan_loss = tfgan.gan_loss(
          gan_model, gradient_penalty_weight=1.0, add_summaries=True)

    # Get the GANTrain ops using the custom optimizers and optional
    # discriminator weight clipping.
    with tf.compat.v1.name_scope('train'):
      gen_opt, dis_opt = _get_optimizers(hparams)
      train_ops = tfgan.gan_train_ops(
          gan_model,
          gan_loss,
          generator_optimizer=gen_opt,
          discriminator_optimizer=dis_opt,
          summarize_gradients=True)

    # Run the alternating training loop. Skip it if no steps should be taken
    # (used for graph construction tests).
    status_message = tf.strings.join([
        'Starting train step: ',
        tf.as_string(tf.compat.v1.train.get_or_create_global_step())
    ],
                                     name='status_message')
    if hparams.max_number_of_steps == 0:
      return
    tfgan.gan_train(
        train_ops,
        hooks=([
            tf.estimator.StopAtStepHook(num_steps=hparams.max_number_of_steps),
            tf.estimator.LoggingTensorHook([status_message], every_n_iter=10)
        ]),
        logdir=hparams.train_log_dir,
        master=hparams.master,
        is_chief=hparams.task == 0)


def _get_optimizers(hparams):
  """Get optimizers that are optionally synchronous."""
  gen_opt = tf.compat.v1.train.AdamOptimizer(hparams.generator_lr, 0.5)
  dis_opt = tf.compat.v1.train.AdamOptimizer(hparams.discriminator_lr, 0.5)

  return gen_opt, dis_opt
