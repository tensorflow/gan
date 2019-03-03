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

from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.cifar import data_provider
from tensorflow_gan.examples.cifar import networks

# ML Hparams.
flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')
flags.DEFINE_integer('max_number_of_steps', 1000000,
                     'The maximum number of gradient steps.')
flags.DEFINE_float('generator_lr', 0.0002, 'The generator learning rate.')
flags.DEFINE_float('discriminator_lr', 0.0002,
                   'The discriminator learning rate.')

# ML Infrastructure.
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_string('train_log_dir', '/tmp/cifar/',
                    'Directory where to write event logs.')
flags.DEFINE_integer(
    'ps_replicas', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

FLAGS = flags.FLAGS


def main(_):
  if not tf.io.gfile.exists(FLAGS.train_log_dir):
    tf.io.gfile.makedirs(FLAGS.train_log_dir)

  with tf.device(tf.compat.v1.train.replica_device_setter(FLAGS.ps_replicas)):
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.compat.v1.name_scope('inputs'):
      with tf.device('/cpu:0'):
        images, _ = data_provider.provide_data(
            'train', FLAGS.batch_size, num_parallel_calls=4)

    # Define the GANModel tuple.
    generator_fn = networks.generator
    discriminator_fn = networks.discriminator
    generator_inputs = tf.random.normal([FLAGS.batch_size, 64])
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
      gen_opt, dis_opt = _get_optimizers()
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
    if FLAGS.max_number_of_steps == 0:
      return
    tfgan.gan_train(
        train_ops,
        hooks=([
            tf.estimator.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
            tf.estimator.LoggingTensorHook([status_message], every_n_iter=10)
        ]),
        logdir=FLAGS.train_log_dir,
        master=FLAGS.master,
        is_chief=FLAGS.task == 0)


def _get_optimizers():
  """Get optimizers that are optionally synchronous."""
  gen_opt = tf.compat.v1.train.AdamOptimizer(FLAGS.generator_lr, 0.5)
  dis_opt = tf.compat.v1.train.AdamOptimizer(FLAGS.discriminator_lr, 0.5)

  return gen_opt, dis_opt


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.app.run()
