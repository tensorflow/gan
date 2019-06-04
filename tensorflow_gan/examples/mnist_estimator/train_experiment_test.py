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

"""Tests for mnist_estimator.train_experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow_gan.examples.mnist_estimator import train_experiment_lib

mock = tf.compat.v1.test.mock


class TrainEstimatorTest(tf.test.TestCase):

  @mock.patch.object(train_experiment_lib.util, 'mnist_score', autospec=True)
  @mock.patch.object(
      train_experiment_lib.util, 'mnist_frechet_distance', autospec=True)
  def test_full_flow(self, mock_mnist_frechet_distance, mock_mnist_score):
    hparams = train_experiment_lib.HParams(
        generator_lr=0.000076421,
        discriminator_lr=0.0031938,
        joint_train=False,
        batch_size=16,
        noise_dims=4,
        model_dir=self.get_temp_dir(),
        num_train_steps=1,
        num_eval_steps=1,
        num_reader_parallel_calls=4,
        use_dummy_data=True)

    # Mock computationally expensive eval computations.
    mock_mnist_score.return_value = 0.0
    mock_mnist_frechet_distance.return_value = 1.0

    train_experiment_lib.train(hparams)

    # Check that there's a .png file in the output directory.
    out_dir = os.path.join(hparams.model_dir, 'outputs')
    self.assertTrue(tf.io.gfile.exists(out_dir))
    has_png = False
    for f in tf.io.gfile.listdir(out_dir):
      if f.split('.')[-1] == 'png':
        has_png = True
        break
    self.assertTrue(has_png)

if __name__ == '__main__':
  tf.test.main()
