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

"""Tests for the generator and its helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_gan.examples.self_attention_estimator import generator


class GeneratorTest(tf.test.TestCase):

  def test_generator_shapes_and_ranges(self):
    """Tests the generator.

    Make sure the image shapes and pixel value ranges are as expected.
    """
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    batch_size = 10
    num_classes = 1000
    zs = tf.random.normal((batch_size, 128))
    gen_class_logits = tf.zeros((batch_size, num_classes))
    gen_class_ints = tf.random.categorical(
        logits=gen_class_logits, num_samples=1)
    gen_sparse_class = tf.squeeze(gen_class_ints)
    images, var_list = generator.generator(
        zs, gen_sparse_class, gf_dim=32, num_classes=num_classes)
    sess = tf.compat.v1.train.MonitoredTrainingSession()
    images_np = sess.run(images)
    self.assertEqual((batch_size, 128, 128, 3), images_np.shape)
    self.assertAllInRange(images_np, -1.0, 1.0)
    self.assertIsInstance(var_list, list)

  def test_usample_shapes(self):
    """Tests that upsampling has the desired effect on shape."""
    image = tf.random.normal([10, 32, 32, 3])
    big_image = generator.usample(image)
    self.assertEqual([10, 64, 64, 3], big_image.shape.as_list())

  def test_upsample_value(self):
    image = tf.random.normal([10, 32, 32, 3])
    big_image = generator.usample(image)
    expected_image = tf.compat.v1.image.resize_nearest_neighbor(
        image, [32 * 2, 32 * 2])
    with self.cached_session() as sess:
      big_image_np, expected_image_np = sess.run([big_image, expected_image])
    self.assertAllEqual(big_image_np, expected_image_np)

  def test_usample_shapes_value_placeholder(self):
    """Tests usample with partially shaped inputs."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    image = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 3])
    big_image = generator.usample(image)
    self.assertEqual([None, 64, 64, 3], big_image.shape.as_list())
    expected_image = tf.compat.v1.image.resize_nearest_neighbor(
        image, [32 * 2, 32 * 2])
    with self.cached_session() as sess:
      big_image_np, expected_image_np = sess.run(
          [big_image, expected_image],
          {image: np.random.normal(size=[10, 32, 32, 3])})
    self.assertAllEqual(big_image_np, expected_image_np)

  def test_block_shapes(self):
    """Tests that passing volumes through blocks affects shapes correctly."""
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    image = tf.random.normal([10, 32, 32, 3])
    label = tf.ones([10,], dtype=tf.int32)
    image_after_block = generator.block(image, label, 13, 1000, 'test_block')
    self.assertEqual([10, 64, 64, 13], image_after_block.shape.as_list())

  def test_make_z_normal(self):
    """Tests the function that makes the latent variable tensors."""
    if tf.executing_eagerly():
      return
    zs = generator.make_z_normal(2, 16, 128)
    sess = tf.compat.v1.train.MonitoredTrainingSession()
    z_batch = sess.run(zs)
    self.assertEqual((2, 16, 128), z_batch.shape)


if __name__ == '__main__':
  tf.test.main()
