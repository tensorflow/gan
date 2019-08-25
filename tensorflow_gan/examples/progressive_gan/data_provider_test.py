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

# python2 python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_gan.examples.progressive_gan import data_provider

mock = tf.compat.v1.test.mock


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


class DataProviderTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DataProviderTest, self).setUp()
    self.testdata_dir = os.path.join(
        flags.FLAGS.test_srcdir,
        'tensorflow_gan/examples/progressive_gan/'
        'testdata/')
    mock_imgs = np.zeros([32, 32, 3], dtype=np.uint8)
    mock_lbls = np.ones([], dtype=np.int64)
    self.mock_ds = tf.data.Dataset.from_tensors({
        'image': mock_imgs,
        'label': mock_lbls
    })

  @mock.patch.object(data_provider, 'tfds', autospec=True)
  def test_provide_dataset(self, mock_tfds):
    batch_size = 4
    patch_height = 2
    patch_width = 8
    colors = 1
    expected_shape = [batch_size, patch_height, patch_width, colors]
    mock_tfds.load.return_value = self.mock_ds

    ds = data_provider.provide_dataset(
        'train',
        patch_height=patch_height,
        patch_width=patch_width,
        colors=colors,
        batch_size=batch_size)
    self.assertIsInstance(ds, tf.data.Dataset)

    output = tf.compat.v1.data.get_output_classes(ds)
    self.assertIsInstance(output, dict)
    self.assertSetEqual(set(output.keys()), set(['images']))
    self.assertEqual(output['images'], tf.Tensor)

    shapes = tf.compat.v1.data.get_output_shapes(ds)
    self.assertIsInstance(shapes, dict)
    self.assertSetEqual(set(shapes.keys()), set(['images']))
    self.assertListEqual(shapes['images'].as_list(), expected_shape)

    types = tf.compat.v1.data.get_output_types(ds)
    self.assertIsInstance(types, dict)
    self.assertSetEqual(set(types.keys()), set(['images']))
    self.assertEqual(types['images'], tf.float32)

    next_batch = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    images = next_batch['images']

    with self.cached_session() as sess:
      images_np = sess.run(images)

    self.assertEqual(images_np.shape, tuple(expected_shape))
    self.assertTrue(np.all(np.abs(images_np) <= 1))

  @mock.patch.object(data_provider, 'tfds', autospec=True)
  def test_provide_data(self, mock_tfds):
    batch_size = 4
    patch_height = 2
    patch_width = 8
    colors = 1
    expected_shape = [batch_size, patch_height, patch_width, colors]
    mock_tfds.load.return_value = self.mock_ds

    images = data_provider.provide_data(
        'train',
        patch_height=patch_height,
        patch_width=patch_width,
        colors=colors,
        batch_size=batch_size)
    self.assertEqual(images.shape.as_list(), expected_shape)

    with self.cached_session() as sess:
      images_np = sess.run(images)

    self.assertTupleEqual(images_np.shape, tuple(expected_shape))
    self.assertTrue(np.all(np.abs(images_np) <= 1))

  @parameterized.parameters(
      {'single_pattern': True},
      {'single_pattern': False},
  )
  def test_provide_data_from_image_files(self, single_pattern):
    batch_size = 2
    patch_height = 3
    patch_width = 4
    colors = 1
    expected_shape = [batch_size, patch_height, patch_width, colors]
    if single_pattern:
      file_pattern = os.path.join(self.testdata_dir, '*.jpg')
    else:
      file_pattern = [os.path.join(self.testdata_dir, '*.jpg')]

    images = data_provider.provide_data_from_image_files(
        file_pattern=file_pattern,
        batch_size=batch_size,
        shuffle=False,
        patch_height=patch_height,
        patch_width=patch_width,
        colors=colors)
    self.assertEqual(images.shape.as_list(), expected_shape)

    with self.cached_session() as sess:
      images_np = sess.run(images)
    self.assertEqual(images_np.shape, tuple(expected_shape))
    self.assertTrue(np.all(np.abs(images_np) <= 1))


if __name__ == '__main__':
  tf.test.main()
