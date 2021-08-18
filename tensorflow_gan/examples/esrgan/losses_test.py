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

"""Tests for tfgan.examples.esrgan.losses"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import os

import tensorflow as tf
import losses 

class LossesTest(tf.test.TestCase, absltest.TestCase):
  def setUp(self):
    super(LossesTest, self).setUp()
    self.real_data = tf.constant([[3.1, 2.3, -12.3, 32.1]])
    self.generated_data = tf.constant([[-12.3, 23.2, 16.3, -43.2]])
    
    self._discriminator_gen_logits = tf.constant([10.0, 4.4, -5.5, 3.6])
    self._discriminator_real_logits = tf.constant([-2.0, 0.4, 12.5, 2.7])
    
    self._expected_pixel_loss = 35.050003
    self._expected_g_loss = 4.9401135 
    self._expected_d_loss = 4.390114
    
  def test_pixel_loss(self):
    pixel_loss = losses.pixel_loss(self.real_data,
                                   self.generated_data)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertNear(self._expected_pixel_loss, 
                      sess.run(pixel_loss), 1e-5)

  def test_ragan_loss(self):
    g_loss = losses.ragan_generator_loss(self._discriminator_real_logits, 
                                         self._discriminator_gen_logits)
    d_loss = losses.ragan_discriminator_loss(self._discriminator_real_logits, 
                                             self._discriminator_gen_logits)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertNear(self._expected_g_loss, sess.run(g_loss), 1e-5)
      self.assertNear(self._expected_d_loss, sess.run(d_loss), 1e-5)

if __name__ == '__main__':
  tf.test.main()
