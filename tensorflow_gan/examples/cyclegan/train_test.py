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

"""Tests for cyclegan.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.cyclegan import train_lib

mock = tf.compat.v1.test.mock


def _test_generator(input_images):
  """Simple generator function."""
  return input_images * tf.compat.v1.get_variable('dummy_g', initializer=2.0)


def _test_discriminator(image_batch, unused_conditioning=None):
  """Simple discriminator function."""
  return tf.compat.v1.layers.flatten(
      image_batch * tf.compat.v1.get_variable('dummy_d', initializer=2.0))


class TrainTest(tf.test.TestCase):

  def setUp(self):
    super(TrainTest, self).setUp()
    self._original_generator = train_lib.networks.generator
    self._original_discriminator = train_lib.networks.discriminator
    train_lib.networks.generator = _test_generator
    train_lib.networks.discriminator = _test_discriminator
    self.hparams = train_lib.HParams(
        image_set_x_file_pattern=None,
        image_set_y_file_pattern=None,
        batch_size=1,
        patch_size=64,
        master='',
        train_log_dir='/tmp/tfgan_logdir/cyclegan/',
        generator_lr=0.0002,
        discriminator_lr=0.0001,
        max_number_of_steps=500000,
        ps_replicas=0,
        task=0,
        cycle_consistency_loss_weight=10.0)

  def tearDown(self):
    super(TrainTest, self).tearDown()
    train_lib.networks.generator = self._original_generator
    train_lib.networks.discriminator = self._original_discriminator

  @mock.patch.object(tfgan, 'eval', autospec=True)
  def test_define_model(self, mock_eval):
    if tf.executing_eagerly():
      # `tfgan.cyclegan_model` doesn't work when executing eagerly.
      return
    self.hparams = self.hparams._replace(batch_size=2)
    images_shape = [self.hparams.batch_size, 4, 4, 3]
    images_x_np = np.zeros(shape=images_shape)
    images_y_np = np.zeros(shape=images_shape)
    images_x = tf.constant(images_x_np, dtype=tf.float32)
    images_y = tf.constant(images_y_np, dtype=tf.float32)

    cyclegan_model = train_lib._define_model(images_x, images_y)
    self.assertIsInstance(cyclegan_model, tfgan.CycleGANModel)
    self.assertShapeEqual(images_x_np, cyclegan_model.reconstructed_x)
    self.assertShapeEqual(images_y_np, cyclegan_model.reconstructed_y)

  @mock.patch.object(train_lib.networks, 'generator', autospec=True)
  @mock.patch.object(train_lib.networks, 'discriminator', autospec=True)
  @mock.patch.object(
      tf.compat.v1.train, 'get_or_create_global_step', autospec=True)
  def test_get_lr(self, mock_get_or_create_global_step,
                  unused_mock_discriminator, unused_mock_generator):
    if tf.executing_eagerly():
      return
    base_lr = 0.01
    max_number_of_steps = 10
    with self.cached_session(use_gpu=True) as sess:
      mock_get_or_create_global_step.return_value = tf.constant(2)
      lr_step2 = sess.run(train_lib._get_lr(base_lr, max_number_of_steps))
      mock_get_or_create_global_step.return_value = tf.constant(9)
      lr_step9 = sess.run(train_lib._get_lr(base_lr, max_number_of_steps))

    self.assertAlmostEqual(base_lr, lr_step2)
    self.assertAlmostEqual(base_lr * 0.2, lr_step9)

  @mock.patch.object(tf.compat.v1.train, 'AdamOptimizer', autospec=True)
  def test_get_optimizer(self, mock_adam_optimizer):
    gen_lr, dis_lr = 0.1, 0.01
    train_lib._get_optimizer(gen_lr=gen_lr, dis_lr=dis_lr)
    mock_adam_optimizer.assert_has_calls([
        mock.call(gen_lr, beta1=mock.ANY, use_locking=True),
        mock.call(dis_lr, beta1=mock.ANY, use_locking=True)
    ])

  def test_define_train_ops(self):
    if tf.executing_eagerly():
      # `tfgan.cyclegan_model` doesn't work when executing eagerly.
      return
    self.hparams = self.hparams._replace(
        batch_size=2, generator_lr=0.1, discriminator_lr=0.01)

    images_shape = [self.hparams.batch_size, 4, 4, 3]
    images_x = tf.zeros(images_shape, dtype=tf.float32)
    images_y = tf.zeros(images_shape, dtype=tf.float32)

    cyclegan_model = train_lib._define_model(images_x, images_y)
    cyclegan_loss = tfgan.cyclegan_loss(
        cyclegan_model, cycle_consistency_loss_weight=10.0)

    train_ops = train_lib._define_train_ops(cyclegan_model, cyclegan_loss,
                                            self.hparams)
    self.assertIsInstance(train_ops, tfgan.GANTrainOps)

  @mock.patch.object(tf.io, 'gfile', autospec=True)
  @mock.patch.object(train_lib, 'data_provider', autospec=True)
  @mock.patch.object(train_lib, '_define_model', autospec=True)
  @mock.patch.object(tfgan, 'cyclegan_loss', autospec=True)
  @mock.patch.object(train_lib, '_define_train_ops', autospec=True)
  @mock.patch.object(tfgan, 'gan_train', autospec=True)
  def test_main(self, mock_gan_train, mock_define_train_ops, mock_cyclegan_loss,
                mock_define_model, mock_data_provider, mock_gfile):
    self.hparams = self.hparams._replace(
        image_set_x_file_pattern='/tmp/x/*.jpg',
        image_set_y_file_pattern='/tmp/y/*.jpg',
        batch_size=3,
        patch_size=8,
        generator_lr=0.02,
        discriminator_lr=0.3,
        train_log_dir='/tmp/foo',
        master='master',
        task=0,
        cycle_consistency_loss_weight=2.0,
        max_number_of_steps=1)

    mock_data_provider.provide_custom_data.return_value = (tf.zeros(
        [3, 2, 2, 3], dtype=tf.float32), tf.zeros([3, 2, 2, 3],
                                                  dtype=tf.float32))

    train_lib.train(self.hparams)
    mock_data_provider.provide_custom_data.assert_called_once_with(
        batch_size=3, image_file_patterns=['/tmp/x/*.jpg', '/tmp/y/*.jpg'],
        patch_size=8)
    mock_define_model.assert_called_once_with(mock.ANY, mock.ANY)
    mock_cyclegan_loss.assert_called_once_with(
        mock_define_model.return_value,
        cycle_consistency_loss_weight=2.0,
        tensor_pool_fn=mock.ANY)
    mock_define_train_ops.assert_called_once_with(
        mock_define_model.return_value, mock_cyclegan_loss.return_value,
        self.hparams)
    mock_gan_train.assert_called_once_with(
        mock_define_train_ops.return_value,
        '/tmp/foo',
        get_hooks_fn=mock.ANY,
        hooks=mock.ANY,
        master='master',
        is_chief=True)


if __name__ == '__main__':
  tf.test.main()
