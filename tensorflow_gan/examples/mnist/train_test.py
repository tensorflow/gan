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

"""Tests for mnist.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf

from tensorflow_gan.examples.mnist import train_lib

mock = tf.compat.v1.test.mock


BATCH_SIZE = 5


def _new_data(*args, **kwargs):
  del args, kwargs
  # Tensors need to be created in the same graph, so generate them at the call
  # site.
  # Note: Make sure batch size matches hparams.
  imgs = tf.zeros([BATCH_SIZE, 28, 28, 1], dtype=tf.float32)
  labels = tf.one_hot([0] * BATCH_SIZE, depth=10)
  return (imgs, labels)


class TrainTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TrainTest, self).setUp()
    self.hparams = train_lib.HParams(
        batch_size=BATCH_SIZE,
        train_log_dir=self.get_temp_dir(),
        max_number_of_steps=1,
        gan_type='unconditional',
        grid_size=1,
        noise_dims=64)

  @mock.patch.object(train_lib.data_provider, 'provide_data', new=_new_data)
  def test_run_one_train_step(self):
    if tf.executing_eagerly():
      # `tfgan.gan_model` doesn't work when executing eagerly.
      return
    train_lib.train(self.hparams)

  @parameterized.parameters(
      {'gan_type': 'unconditional'},
      {'gan_type': 'conditional'},
      {'gan_type': 'infogan'},
  )
  @mock.patch.object(train_lib.data_provider, 'provide_data', new=_new_data)
  def test_build_graph(self, gan_type):
    if tf.executing_eagerly():
      # `tfgan.gan_model` doesn't work when executing eagerly.
      return
    hparams = self.hparams._replace(max_number_of_steps=0, gan_type=gan_type)
    train_lib.train(hparams)


if __name__ == '__main__':
  tf.test.main()
