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

"""Tests for tfgan.features.conditioning_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_gan as tfgan


class ConditioningUtilsTest(tf.test.TestCase):

  def test_condition_tensor_multiple_shapes(self):
    for tensor_shape in [(4, 1), (4, 2), (4, 2, 6), (None, 5, 3)]:
      for conditioning_shape in [(4, 1), (4, 8), (4, 5, 3)]:
        tfgan.features.condition_tensor(
            tf.placeholder(tf.float32, tensor_shape),
            tf.placeholder(tf.float32, conditioning_shape))

  def test_condition_tensor_asserts(self):
    with self.assertRaisesRegexp(ValueError, 'Cannot reshape'):
      tfgan.features.condition_tensor(
          tf.placeholder(tf.float32, (4, 1)),
          tf.placeholder(tf.float32, (5, 1)))

    with self.assertRaisesRegexp(ValueError, 'Shape .* is not fully defined'):
      tfgan.features.condition_tensor(
          tf.placeholder(tf.float32, (5, None)),
          tf.placeholder(tf.float32, (5, 1)))

    with self.assertRaisesRegexp(ValueError, 'at least 2D'):
      tfgan.features.condition_tensor(
          tf.placeholder(tf.float32, (5, 2)),
          tf.placeholder(tf.float32, (5)))

  def test_condition_tensor_from_onehot(self):
    tfgan.features.condition_tensor_from_onehot(
        tf.placeholder(tf.float32, (5, 4, 1)),
        tf.placeholder(tf.float32, (5, 10)))

  def test_condition_tensor_from_onehot_asserts(self):
    with self.assertRaisesRegexp(ValueError, 'Shape .* must have rank 2'):
      tfgan.features.condition_tensor_from_onehot(
          tf.placeholder(tf.float32, (5, 1)),
          tf.placeholder(tf.float32, (5)))

    with self.assertRaisesRegexp(ValueError, 'Shape .* is not fully defined'):
      tfgan.features.condition_tensor_from_onehot(
          tf.placeholder(tf.float32, (5, 1)),
          tf.placeholder(tf.float32, (5, None)))

    with self.assertRaisesRegexp(ValueError, 'Cannot reshape a tensor'):
      tfgan.features.condition_tensor_from_onehot(
          tf.placeholder(tf.float32, (5, 1)),
          tf.placeholder(tf.float32, (4, 6)))


if __name__ == '__main__':
  tf.test.main()
