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

"""Tests for discriminator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_gan.examples.cyclegan import discriminator


class DiscriminatorTest(tf.test.TestCase):

  def _layer_output_size(self, input_size, kernel_size=4, stride=2, pad=2):
    return (input_size + pad * 2 - kernel_size) // stride + 1

  def test_four_layers(self):
    batch_size = 2
    input_size = 256

    output_size = self._layer_output_size(input_size)
    output_size = self._layer_output_size(output_size)
    output_size = self._layer_output_size(output_size)
    output_size = self._layer_output_size(output_size, stride=1)
    output_size = self._layer_output_size(output_size, stride=1)

    images = tf.ones((batch_size, input_size, input_size, 3))
    logits, end_points = discriminator.pix2pix_discriminator(
        images, num_filters=[64, 128, 256, 512])
    self.assertListEqual([batch_size, output_size, output_size, 1],
                         logits.shape.as_list())
    self.assertListEqual([batch_size, output_size, output_size, 1],
                         end_points['predictions'].shape.as_list())

  def test_four_layers_no_padding(self):
    batch_size = 2
    input_size = 256

    output_size = self._layer_output_size(input_size, pad=0)
    output_size = self._layer_output_size(output_size, pad=0)
    output_size = self._layer_output_size(output_size, pad=0)
    output_size = self._layer_output_size(output_size, stride=1, pad=0)
    output_size = self._layer_output_size(output_size, stride=1, pad=0)

    images = tf.ones((batch_size, input_size, input_size, 3))
    logits, end_points = discriminator.pix2pix_discriminator(
        images, num_filters=[64, 128, 256, 512], padding=0)
    self.assertListEqual([batch_size, output_size, output_size, 1],
                         logits.shape.as_list())
    self.assertListEqual([batch_size, output_size, output_size, 1],
                         end_points['predictions'].shape.as_list())

  def test_four_layers_wrong_paddig(self):
    batch_size = 2
    input_size = 256

    images = tf.ones((batch_size, input_size, input_size, 3))
    with self.assertRaises(TypeError):
      discriminator.pix2pix_discriminator(
          images, num_filters=[64, 128, 256, 512], padding=1.5)

  def test_four_layers_negative_padding(self):
    batch_size = 2
    input_size = 256

    images = tf.ones((batch_size, input_size, input_size, 3))
    if tf.executing_eagerly():
      exception_type = tf.errors.InvalidArgumentError
    else:
      exception_type = ValueError
    with self.assertRaises(exception_type):
      discriminator.pix2pix_discriminator(
          images, num_filters=[64, 128, 256, 512], padding=-1)

if __name__ == '__main__':
  tf.test.main()
