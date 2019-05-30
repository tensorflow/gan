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

from absl import flags
from absl.testing import parameterized
import numpy as np

import tensorflow as tf

from tensorflow_gan.examples.mnist import train

FLAGS = flags.FLAGS
mock = tf.compat.v1.test.mock


class TrainTest(tf.test.TestCase, parameterized.TestCase):

  @mock.patch.object(train, 'data_provider', autospec=True)
  def test_run_one_train_step(self, mock_data_provider):
    FLAGS.max_number_of_steps_mnist = 1
    FLAGS.gan_type = 'unconditional'
    FLAGS.batch_size_mnist = 5
    FLAGS.grid_size = 1
    tf.compat.v1.set_random_seed(1234)

    # Mock input pipeline.
    mock_imgs = np.zeros([FLAGS.batch_size_mnist, 28, 28, 1], dtype=np.float32)
    mock_lbls = np.concatenate(
        (np.ones([FLAGS.batch_size_mnist, 1], dtype=np.int32),
         np.zeros([FLAGS.batch_size_mnist, 9], dtype=np.int32)),
        axis=1)
    mock_data_provider.provide_data.return_value = (mock_imgs, mock_lbls)

    train.main(None)

  @parameterized.named_parameters(('Unconditional', 'unconditional'),
                                  ('Conditional', 'conditional'),
                                  ('InfoGAN', 'infogan'))
  @mock.patch.object(train, 'data_provider', autospec=True)
  def test_build_graph(self, gan_type, mock_data_provider):
    FLAGS.max_number_of_steps_mnist = 0
    FLAGS.gan_type = gan_type

    # Mock input pipeline.
    mock_imgs = np.zeros([FLAGS.batch_size_mnist, 28, 28, 1], dtype=np.float32)
    mock_lbls = np.concatenate(
        (np.ones([FLAGS.batch_size_mnist, 1], dtype=np.int32),
         np.zeros([FLAGS.batch_size_mnist, 9], dtype=np.int32)),
        axis=1)
    mock_data_provider.provide_data.return_value = (mock_imgs, mock_lbls)

    train.main(None)


if __name__ == '__main__':
  tf.test.main()
