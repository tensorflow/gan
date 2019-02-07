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

"""Tests for tfgan.eval.tfgan.eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan


class UtilsTest(tf.test.TestCase):

  def test_image_grid(self):
    tfgan.eval.image_grid(
        input_tensor=tf.zeros([25, 32, 32, 3]),
        grid_shape=(5, 5))

  def test_python_image_grid(self):
    image_grid = tfgan.eval.python_image_grid(
        input_array=np.zeros([25, 32, 32, 3]),
        grid_shape=(5, 5))
    self.assertTupleEqual(image_grid.shape, (5 * 32, 5 * 32, 3))

  # TODO(joelshor): Add more `image_reshaper` tests.
  def test_image_reshaper_image_list(self):
    images = tfgan.eval.image_reshaper(
        images=tf.unstack(tf.zeros([25, 32, 32, 3])),
        num_cols=2)
    images.shape.assert_is_compatible_with([1, 13 * 32, 2 * 32, 3])

  def test_image_reshaper_image(self):
    images = tfgan.eval.image_reshaper(
        images=tf.zeros([25, 32, 32, 3]),
        num_cols=2)
    images.shape.assert_is_compatible_with([1, 13 * 32, 2 * 32, 3])


if __name__ == '__main__':
  tf.test.main()
