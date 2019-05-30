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

"""Tests for CIFAR10 networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
from tensorflow_gan.examples.cifar import networks


class NetworksTest(tf.test.TestCase):

  def test_generator(self):
    tf.compat.v1.set_random_seed(1234)
    batch_size = 100
    noise = tf.random.normal([batch_size, 64])
    image = networks.generator(noise)
    with self.cached_session(use_gpu=True) as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      image_np = sess.run(image)

    self.assertAllEqual([batch_size, 32, 32, 3], image_np.shape)
    self.assertTrue(np.all(np.abs(image_np) <= 1))

  def test_discriminator(self):
    batch_size = 5
    image = tf.random.uniform([batch_size, 32, 32, 3], -1, 1)
    dis_output = networks.discriminator(image, None)
    with self.cached_session(use_gpu=True) as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      dis_output_np = sess.run(dis_output)

    self.assertAllEqual([batch_size, 1], dis_output_np.shape)


if __name__ == '__main__':
  tf.test.main()
