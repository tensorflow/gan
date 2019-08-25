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

"""Tests for data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_gan.examples.cifar import data_provider

mock = tf.compat.v1.test.mock


class DataProviderTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DataProviderTest, self).setUp()
    mock_imgs = np.zeros([32, 32, 3], dtype=np.uint8)
    mock_lbls = np.ones([], dtype=np.int64)
    self.mock_ds = tf.data.Dataset.from_tensors({
        'image': mock_imgs,
        'label': mock_lbls
    })

  @parameterized.parameters(
      {'one_hot': True},
      {'one_hot': False},
  )
  @mock.patch.object(data_provider, 'tfds', autospec=True)
  def test_provide_dataset(self, mock_tfds, one_hot):
    batch_size = 5
    mock_tfds.load.return_value = self.mock_ds

    ds = data_provider.provide_dataset('test', batch_size, one_hot=one_hot)
    self.assertIsInstance(ds, tf.data.Dataset)

    output = tf.compat.v1.data.get_output_classes(ds)
    self.assertIsInstance(output, dict)
    self.assertSetEqual(set(output.keys()), set(['images', 'labels']))
    self.assertEqual(output['images'], tf.Tensor)
    self.assertEqual(output['labels'], tf.Tensor)

    shapes = tf.compat.v1.data.get_output_shapes(ds)
    self.assertIsInstance(shapes, dict)
    self.assertSetEqual(set(shapes.keys()), set(['images', 'labels']))
    self.assertIsInstance(shapes['images'], tf.TensorShape)
    self.assertListEqual(shapes['images'].as_list(), [batch_size, 32, 32, 3])
    if one_hot:
      expected_lbls_shape = [batch_size, 10]
    else:
      expected_lbls_shape = [batch_size]
    self.assertListEqual(shapes['labels'].as_list(), expected_lbls_shape)

    types = tf.compat.v1.data.get_output_types(ds)
    self.assertIsInstance(types, dict)
    self.assertSetEqual(set(types.keys()), set(['images', 'labels']))
    self.assertEqual(types['images'], tf.float32)
    self.assertEqual(types['labels'], tf.float32)

    next_batch = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    images = next_batch['images']
    labels = next_batch['labels']

    with self.cached_session() as sess:
      images, labels = sess.run([images, labels])

    self.assertEqual(images.shape, (batch_size, 32, 32, 3))
    self.assertTrue(np.all(np.abs(images) <= 1))
    self.assertEqual(labels.shape, tuple(expected_lbls_shape))

  @parameterized.parameters(
      {'one_hot': True},
      {'one_hot': False},
  )
  @mock.patch.object(data_provider, 'tfds', autospec=True)
  def test_provide_data(self, mock_tfds, one_hot):
    batch_size = 5
    mock_tfds.load.return_value = self.mock_ds

    images, labels = data_provider.provide_data(
        'test', batch_size, one_hot=one_hot)

    with self.cached_session() as sess:
      images, labels = sess.run([images, labels])
    self.assertTupleEqual(images.shape, (batch_size, 32, 32, 3))
    self.assertTrue(np.all(np.abs(images) <= 1))
    if one_hot:
      expected_lbls_shape = (batch_size, 10)
    else:
      expected_lbls_shape = (batch_size,)
    self.assertTupleEqual(labels.shape, expected_lbls_shape)

  @mock.patch.object(data_provider, 'tfds', autospec=True)
  def test_provide_data_can_be_reinitialized(self, mock_tfds):
    """Test that the iterator created in `provide_data` can be reused."""
    if tf.executing_eagerly():
      return
    batch_size = 5
    mock_tfds.load.return_value = self.mock_ds

    images, labels = data_provider.provide_data('test', batch_size)

    with self.session() as sess:
      sess.run([images, labels])
      sess.run([images, labels])
    with self.session() as sess:
      sess.run([images, labels])
      sess.run([images, labels])


if __name__ == '__main__':
  tf.test.main()
