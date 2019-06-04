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

"""Tests for stargan_estimator.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_gan.examples.stargan_estimator import train_lib

mock = tf.compat.v1.test.mock


def _test_generator(input_images, _):
  """Simple generator function."""
  return input_images * tf.compat.v1.get_variable('dummy_g', initializer=2.0)


def _test_discriminator(inputs, num_domains):
  """Differentiable dummy discriminator for StarGAN."""
  hidden = tf.compat.v1.layers.flatten(inputs)
  output_src = tf.reduce_mean(input_tensor=hidden, axis=1)
  output_cls = tf.compat.v1.layers.dense(inputs=hidden, units=num_domains)
  return output_src, output_cls


class TrainTest(tf.test.TestCase):

  @mock.patch.object(train_lib.data_provider, 'provide_data', autospec=True)
  @mock.patch.object(
      train_lib.data_provider, 'provide_celeba_test_set', autospec=True)
  def test_main(self, mock_provide_celeba_test_set, mock_provide_data):
    hparams = train_lib.HParams(
        batch_size=1,
        patch_size=8,
        output_dir='/tmp/tfgan_logdir/stargan/',
        generator_lr=1e-4,
        discriminator_lr=1e-4,
        max_number_of_steps=0,
        steps_per_eval=1,
        adam_beta1=0.5,
        adam_beta2=0.999,
        gen_disc_step_ratio=0.2,
        master='',
        ps_tasks=0,
        task=0)
    num_domains = 3

    # Construct mock inputs.
    images_shape = [
        hparams.batch_size, hparams.patch_size, hparams.patch_size, 3
    ]
    img_list = [np.zeros(images_shape, dtype=np.float32)] * num_domains
    # Create a list of num_domains arrays of shape [batch_size, num_domains].
    # Note: assumes hparams.batch_size <= num_domains.
    lbl_list = [np.eye(num_domains)[:hparams.batch_size, :]] * num_domains
    mock_provide_data.return_value = (img_list, lbl_list)
    mock_provide_celeba_test_set.return_value = np.zeros(
        [3, hparams.patch_size, hparams.patch_size, 3])

    train_lib.train(hparams, _test_generator, _test_discriminator)


if __name__ == '__main__':
  tf.test.main()
