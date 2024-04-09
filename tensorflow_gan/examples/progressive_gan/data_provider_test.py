# coding=utf-8
# Copyright 2024 The TensorFlow GAN Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_gan.examples.progressive_gan import data_provider

mock = tf.test.mock


class DataProviderUtilsTest(tf.test.TestCase):

  def test_normalize_image(self):
    image_np = np.asarray([0, 255, 210], dtype=np.uint8)
    normalized_image = data_provider.normalize_image(tf.constant(image_np))

    # Static checks.
    self.assertEqual(normalized_image.dtype, tf.float32)
    self.assertEqual(normalized_image.shape.as_list(), [3])

    # Run the graph and check the result.
    with self.cached_session() as sess:
      normalized_image_np = sess.run(normalized_image)
    self.assertAllClose(normalized_image_np, [-1, 1, 0.6470588235], 1.0e-6)

  def test_sample_patch_large_patch_returns_upscaled_image(self):
    image_np = np.reshape(np.arange(2 * 2), [2, 2, 1])
    image = tf.constant(image_np, dtype=tf.float32)
    image_patch = data_provider.sample_patch(
        image, patch_height=3, patch_width=3, colors=1)
    with self.cached_session() as sess:
      image_patch_np = sess.run(image_patch)
    expected_np = np.asarray([[[0.], [0.66666669], [1.]],
                              [[1.33333337], [2.], [2.33333349]],
                              [[2.], [2.66666675], [3.]]])
    self.assertAllClose(image_patch_np, expected_np, 1.0e-6)

  def test_sample_patch_small_patch_returns_downscaled_image(self):
    image_np = np.reshape(np.arange(3 * 3), [3, 3, 1])
    image = tf.constant(image_np, dtype=tf.float32)
    image_patch = data_provider.sample_patch(
        image, patch_height=2, patch_width=2, colors=1)

    with self.cached_session() as sess:
      image_patch_np = sess.run(image_patch)
    expected_np = np.asarray([[[0.], [1.5]], [[4.5], [6.]]])
    self.assertAllClose(image_patch_np, expected_np, 1.0e-6)




if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
