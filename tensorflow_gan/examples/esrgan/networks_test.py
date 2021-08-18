# coding=utf-8
# Copyright 2021 The TensorFlow GAN Authors.
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

"""Tests for tfgan.examples.esrgan.networks"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import networks
import collections

Params = collections.namedtuple('HParams', ['hr_dimension', 
                                            'scale', 
                                            'trunk_size'])
class NetworksTest(tf.test.TestCase):
  def setUp(self):
    self.HParams = Params(256, 4, 11)
    self.generator = networks.generator_network(self.HParams)
    self.discriminator = networks.discriminator_network(self.HParams)

  def test_network_type(self):
    """ Verifies that the models are of keras.Model type """
    self.assertIsInstance(self.generator, tf.keras.Model)
    self.assertIsInstance(self.discriminator, tf.keras.Model)

  def test_generator(self):
    """ Verifies the generator output shape."""
    img_batch = tf.random.uniform([3, 64, 64, 3])
    target_shape = [3, 256, 256, 3]
    model_output = self.generator(img_batch)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertEqual(model_output.shape, target_shape)
  
  def test_generator_inference(self):
    """ Check one inference step."""
    img_batch = tf.zeros([2, 64, 64, 3])
    model_output, _ = self.generator(img_batch)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(model_output)

  def test_discriminator(self):
    img_batch = tf.zeros([3, 256, 256, 3])
    disc_output = self.discriminator(img_batch)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(disc_output)


if __name__ == '__main__':
  tf.test.main()
