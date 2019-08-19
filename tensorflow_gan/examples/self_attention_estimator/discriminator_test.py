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

"""Tests for the discriminator and its helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_gan.examples.self_attention_estimator import discriminator


class DiscriminatorTest(tf.test.TestCase):

  def test_generator_shapes_and_ranges(self):
    """Tests the discriminator.

    Make sure the image shapes and output value ranges are as expected.
    """
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    batch_size = 10
    num_classes = 1000
    gen_class_logits = tf.zeros((batch_size, num_classes))
    gen_class_ints = tf.random.categorical(
        logits=gen_class_logits, num_samples=1)
    gen_sparse_class = tf.squeeze(gen_class_ints)
    images = tf.random.normal([10, 32, 32, 3])
    d_out, var_list = discriminator.discriminator(images, gen_sparse_class, 16,
                                                  1000)
    sess = tf.compat.v1.train.MonitoredTrainingSession()
    images_np = sess.run(d_out)
    self.assertEqual((batch_size, 1), images_np.shape)
    self.assertAllInRange(images_np, -1.0, 1.0)
    self.assertIsInstance(var_list, list)

  def test_dsample_shapes(self):
    """Tests that downsampling has the desired effect on shape."""
    image = tf.random.normal([10, 32, 32, 3])
    big_image = discriminator.dsample(image)
    self.assertEqual([10, 16, 16, 3], big_image.shape.as_list())

  def test_block_shapes(self):
    """Tests that passing volumes through blocks affects shapes correctly."""
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    image = tf.random.normal([10, 32, 32, 3])
    image_after_block = discriminator.block(image, 13, 'test_block')
    self.assertEqual([10, 16, 16, 13], image_after_block.shape.as_list())

  def test_optimized_block_shapes(self):
    """Tests that passing volumes through blocks affects shapes correctly."""
    if tf.executing_eagerly():
      #  `compute_spectral_norm` doesn't work when executing eagerly.
      return
    image = tf.random.normal([10, 32, 32, 3])
    image_after_block = discriminator.optimized_block(image, 13, 'test_block')
    self.assertEqual([10, 16, 16, 13], image_after_block.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
