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

"""Tests for generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_gan.examples.cyclegan import generator


# TODO(joelshor): Add a test to check generator endpoints.
class CycleganTest(tf.test.TestCase, parameterized.TestCase):

  def test_generator_inference(self):
    """Check one inference step."""
    img_batch = tf.zeros([2, 32, 32, 3])
    model_output, _ = generator.cyclegan_generator_resnet(img_batch)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(model_output)

  @parameterized.parameters(
      {'shape': [4, 32, 32, 3]},  # small
      {'shape': [3, 128, 128, 3]},  # medium
      {'shape': [2, 80, 400, 3]},  # nonsquare
  )
  def test_generator_graph(self, shape):
    """Check that generator can take small and non-square inputs."""
    output_imgs, _ = generator.cyclegan_generator_resnet(tf.ones(shape))
    self.assertAllEqual(shape, output_imgs.shape.as_list())

  def test_generator_unknown_batch_dim(self):
    """Check that generator can take unknown batch dimension inputs."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    img = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, None, 3])
    output_imgs, _ = generator.cyclegan_generator_resnet(img)

    self.assertAllEqual([None, 32, None, 3], output_imgs.shape.as_list())

  @parameterized.parameters(
      {'kernel_size': 3},
      {'kernel_size': 4},
      {'kernel_size': 5},
      {'kernel_size': 6},
  )
  def input_and_output_same_shape(self, kernel_size):
    img_batch = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 3])
    output_img_batch, _ = generator.cyclegan_generator_resnet(
        img_batch, kernel_size=kernel_size)

    self.assertAllEqual(img_batch.shape.as_list(),
                        output_img_batch.shape.as_list())

  @parameterized.parameters(
      {'height': 29},
      {'height': 30},
      {'height': 31},
  )
  def error_if_height_not_multiple_of_four(self, height):
    self.assertRaisesRegexp(
        ValueError,
        'The input height must be a multiple of 4.',
        generator.cyclegan_generator_resnet,
        tf.compat.v1.placeholder(tf.float32, shape=[None, height, 32, 3]))

  @parameterized.parameters(
      {'width': 29},
      {'width': 30},
      {'width': 31},
  )
  def error_if_width_not_multiple_of_four(self, width):
    self.assertRaisesRegexp(
        ValueError,
        'The input width must be a multiple of 4.',
        generator.cyclegan_generator_resnet,
        tf.compat.v1.placeholder(tf.float32, shape=[None, 32, width, 3]))


if __name__ == '__main__':
  tf.test.main()
