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

"""Tests for spectral norm ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_gan.examples.self_attention_estimator import ops


class OpsTest(tf.test.TestCase):

  def test_snconv2d_shapes(self):
    """Tests the spectrally normalized 2d conv function.

    This is a minimal test to make sure that shapes are OK.
    The image shape should match after snconv is applied.
    """
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    image = tf.random.normal([10, 32, 32, 3])
    snconv_image = ops.snconv2d(image, 3, k_h=3, k_w=3, d_h=1, d_w=1)
    self.assertEqual([10, 32, 32, 3], snconv_image.shape.as_list())

  def test_snlinear_shapes(self):
    """Tests the spectrally normalized linear layer.

    This is a minimal test to make sure that shapes are OK.
    The vector shape should match after snlinear.
    """
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    vector = tf.random.normal([10, 32])
    snconv_vector = ops.snlinear(vector, 32)
    self.assertEqual([10, 32], snconv_vector.shape.as_list())

  def test_sn_embedding_shapes(self):
    """Tests the spectrally normalized embedding layer.

    When label = 10, embedding_size = 128, the
    output shape should be [10, 128]
    """
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    label = tf.ones([10,], dtype=tf.int32)
    vector = ops.sn_embedding(label, number_classes=1000, embedding_size=128)
    self.assertEqual([10, 128], vector.shape.as_list())

  def test_conditional_batch_norm_shapes(self):
    """Tests the conditional batch norm layer.

    This is a minimal test to make sure that shapes are OK.
    """
    c_bn = ops.ConditionalBatchNorm(num_categories=1000)
    label = tf.ones([10,], dtype=tf.int32)
    image = tf.random.normal([10, 32, 32, 3])
    bn_image = c_bn(image, label)
    self.assertEqual([10, 32, 32, 3], bn_image.shape.as_list())

  def test_batch_norm_shapes(self):
    """Tests the batch norm layer.

    This is a minimal test to make sure that shapes are OK.
    """
    bn = ops.BatchNorm()
    image = tf.random.normal([10, 32, 32, 3])
    bn_image = bn(image)
    self.assertEqual([10, 32, 32, 3], bn_image.shape.as_list())

  def test_sn_conv1x1_shapes(self):
    """Tests that downsampling has the desired effect on shape."""
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    image = tf.random.normal([10, 32, 32, 3])
    big_image = ops.sn_conv1x1(image, 7, name='test_conv')
    self.assertEqual([10, 32, 32, 7], big_image.shape.as_list())

  def test_sn_non_local_block_sim_shapes(self):
    """Tests that downsampling has the desired effect on shape."""
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    image = tf.random.normal([10, 8, 8, 64])
    big_image = ops.sn_non_local_block_sim(image, name='test_sa')
    self.assertEqual([10, 8, 8, 64], big_image.shape.as_list())

if __name__ == '__main__':
  tf.test.main()
