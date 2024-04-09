# coding=utf-8
# Copyright 2024 The TensorFlow GAN Authors.
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

"""Tests for tfgan.examples.esrgan.train."""

import tensorflow as tf
from tensorflow_gan.examples.esrgan import train_lib

mock = tf.compat.v1.test.mock


class TrainTest(tf.test.TestCase):

  def setUp(self):
    # Parameters for a single step of training.
    super(TrainTest, self).setUp()
    self.hparams = train_lib.HParams(32, 4, '/content/', False, False, 256,
                                     '/content/', 1, 11, 1, 1, 0.5, 0.0001, 0.5,
                                     0.001, 0.00005, 'L1', 0.001, 0.5,
                                     '/content/')
    d = tf.data.Dataset.from_tensor_slices(tf.random.normal([32, 256, 256, 3]))

    def lr(hr):
      lr = tf.image.resize(hr, [64, 64], method='bicubic')
      return lr, hr

    d = d.map(lr)
    d = d.batch(2)
    self.mock_dataset = d

  def test_pretrain_generator(self):
    """Executes all the processes inside the phase-1 training step, once."""
    self.assertIsNone(
        train_lib.pretrain_generator(self.hparams, self.mock_dataset))

  def test_train_generator(self):
    """Executes the phase-2 training step for a single step, once."""
    self.assertIsNone(train_lib.train_esrgan(self.hparams, self.mock_dataset))


if __name__ == '__main__':
  tf.test.main()
