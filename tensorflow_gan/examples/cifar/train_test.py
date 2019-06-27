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

"""Tests for cifar.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_gan.examples.cifar import train_lib

mock = tf.compat.v1.test.mock


class TrainTest(tf.test.TestCase):

  @mock.patch.object(train_lib, 'data_provider', autospec=True)
  def test_build_graph(self, mock_data_provider):
    if tf.executing_eagerly():
      # `tfgan.gan_model` doesn't work when executing eagerly.
      return
    hparams = train_lib.HParams(
        batch_size=16,
        max_number_of_steps=0,
        generator_lr=0.0002,
        discriminator_lr=0.0002,
        master='',
        train_log_dir='/tmp/tfgan_logdir/cifar/',
        ps_replicas=0,
        task=0)

    # Mock input pipeline.
    mock_imgs = np.zeros([hparams.batch_size, 32, 32, 3], dtype=np.float32)
    mock_lbls = np.concatenate(
        (np.ones([hparams.batch_size, 1], dtype=np.float32),
         np.zeros([hparams.batch_size, 9], dtype=np.float32)),
        axis=1)
    mock_data_provider.provide_data.return_value = (mock_imgs, mock_lbls)

    train_lib.train(hparams)


if __name__ == '__main__':
  tf.test.main()
