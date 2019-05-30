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

"""Tests for stargan.data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_gan.examples.stargan import data_provider

mock = tf.compat.v1.test.mock


class DataProviderTest(tf.test.TestCase, absltest.TestCase):

  def setUp(self):
    super(DataProviderTest, self).setUp()
    mock_imgs = np.zeros([128, 128, 3], dtype=np.uint8)
    self.mock_ds = tf.data.Dataset.from_tensors(
        {'attributes': {
            'A': True,
            'B': True,
            'C': True},
         'image': mock_imgs})

  @mock.patch.object(data_provider, 'tfds', autospec=True)
  def test_provide_data(self, mock_tfds):
    batch_size = 5
    patch_size = 32
    mock_tfds.load.return_value = self.mock_ds

    images, labels = data_provider.provide_data(
        'test', batch_size, patch_size=patch_size, domains=('A', 'B', 'C'))
    self.assertLen(images, 3)
    self.assertLen(labels, 3)

    with self.cached_session() as sess:
      images = sess.run(images)
      labels = sess.run(labels)
    for img in images:
      self.assertTupleEqual(img.shape, (batch_size, patch_size, patch_size, 3))
      self.assertTrue(np.all(np.abs(img) <= 1))
    for lbl in labels:
      expected_lbls_shape = (batch_size, 3)
      self.assertTupleEqual(lbl.shape, expected_lbls_shape)


if __name__ == '__main__':
  tf.test.main()
