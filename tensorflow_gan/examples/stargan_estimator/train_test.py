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

from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow_gan.examples.stargan_estimator import train

FLAGS = flags.FLAGS
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

  @mock.patch.object(train.data_provider, 'provide_data', autospec=True)
  @mock.patch.object(train.data_provider, 'provide_celeba_test_set',
                     autospec=True)
  def test_main(self, mock_provide_celeba_test_set, mock_provide_data):
    FLAGS.max_number_of_steps_stargan_estimator = 0
    FLAGS.steps_per_eval = 1
    FLAGS.batch_size_stargan_estimator = 1
    FLAGS.patch_size_stargan_estimator = 8
    num_domains = 3

    # Construct mock inputs.
    images_shape = [FLAGS.batch_size_stargan_estimator, FLAGS.patch_size_stargan_estimator, FLAGS.patch_size_stargan_estimator, 3]
    img_list = [np.zeros(images_shape, dtype=np.float32)] * num_domains
    # Create a list of num_domains arrays of shape [batch_size_stargan_estimator, num_domains].
    # Note: assumes FLAGS.batch_size_stargan_estimator <= num_domains.
    lbl_list = [np.eye(num_domains)[:FLAGS.batch_size_stargan_estimator, :]] * num_domains
    mock_provide_data.return_value = (img_list, lbl_list)
    mock_provide_celeba_test_set.return_value = np.zeros(
        [3, FLAGS.patch_size_stargan_estimator, FLAGS.patch_size_stargan_estimator, 3])

    train.main(None, _test_generator, _test_discriminator)


if __name__ == '__main__':
  tf.test.main()
