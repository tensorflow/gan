# coding=utf-8
# Copyright 2018 The TensorFlow GAN Authors.
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

# Dependency imports
from absl import flags
import tensorflow as tf

from tensorflow_gan.examples.stargan_estimator import train

FLAGS = flags.FLAGS
mock = tf.test.mock


def _test_generator(input_images, _):
  """Simple generator function."""
  return input_images * tf.get_variable('dummy_g', initializer=2.0)


def _test_discriminator(inputs, num_domains):
  """Differentiable dummy discriminator for StarGAN."""
  hidden = tf.contrib.layers.flatten(inputs)
  output_src = tf.reduce_mean(hidden, axis=1)
  output_cls = tf.contrib.layers.fully_connected(
      inputs=hidden,
      num_outputs=num_domains,
      activation_fn=None,
      normalizer_fn=None,
      biases_initializer=None)
  return output_src, output_cls


class TrainTest(tf.test.TestCase):

  @mock.patch.object(train.data_provider, 'provide_data', autospec=True)
  def test_main(self, mock_provide_data):
    FLAGS.max_number_of_steps = 0
    FLAGS.steps_per_eval = 1
    FLAGS.batch_size = 1
    FLAGS.patch_size = 8
    num_domains = 3

    # Construct mock inputs.
    images_shape = [FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, 3]
    img_list = [tf.zeros(images_shape)] * num_domains
    lbl_list = [tf.one_hot([0] * FLAGS.batch_size, num_domains)] * num_domains
    mock_provide_data.return_value = (img_list, lbl_list)

    train.main(None, _test_generator, _test_discriminator)


if __name__ == '__main__':
  tf.test.main()

