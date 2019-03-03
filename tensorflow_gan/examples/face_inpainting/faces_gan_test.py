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

"""Testing faces_gan.

Tests for face_inpainting.faces_gan.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from absl import flags
from absl.testing import flagsaver
import mock
import numpy as np
import tensorflow as tf
from tensorflow_gan.examples.face_inpainting import faces_gan

FLAGS = flags.FLAGS


class FacesGanTest(tf.test.TestCase):

  def test_open_image(self):
    """Make sure open_image returns a resized normalized output."""

    # Patch the decode_png to avoid piping in an actual image. Make sure the
    # input is multiplied by 256 so that the output of the function returns ones
    # everywhere.
    with mock.patch.object(
        faces_gan.tf.image, 'decode_png',
        return_value=tf.fill([100, 100, 3], value=256)):
      # Run open_image with a smaller shape to check the resize. We dont care
      # about the filepath because the file open function is mocked.
      image_tensor = faces_gan.open_image('dummy', imshape=(10, 10))

    # Get the output to compare with an expected numpy array.
    sess = tf.compat.v1.Session()
    image = sess.run(image_tensor)

    # If normalized from [0-256) to [-1-1] correctly, the output should be
    # approximately an array of all 1.0s (tf.image.resize_area changes the
    # output values slightly).
    approximate_equality = np.isclose(image, np.ones([10, 10, 3]), atol=1e-5)
    self.assertTrue(np.all(approximate_equality))

  def test_get_get_batch(self):
    """Make sure get_get_batch pipes input data correctly."""

    batch_size = 16
    z_dim = 10
    imshape = (10, 10)

    # Patch list_files to test the actual dataset pipeline instead of getting in
    # the weeds with loading actual image data from a directory. As such, the
    # data_dir parameter doesnt really matter.
    with mock.patch.object(faces_gan.tf.data.Dataset,
                           'list_files', return_value='dummy/file/path'):
      input_fn = faces_gan.get_get_batch('dummy', batch_size, z_dim, imshape)

    # Make sure we can get data multiple times.
    for _ in range(3):
      # Get the output data.
      test_input = input_fn()
      self.assertLen(test_input, 2)
      noise, image = test_input

      # Check for shapes and batches.
      self.assertSequenceEqual(noise.get_shape().as_list(), (batch_size, z_dim))
      self.assertSequenceEqual(image.get_shape().as_list(),
                               (batch_size, imshape[0], imshape[1], 3))

  @flagsaver.flagsaver
  def test_gan_model(self):
    """Make sure the gan model builds and can train."""

    # Set up variables for the mocked test.
    FLAGS.num_steps = 2
    FLAGS.add_summaries = False
    FLAGS.save_dir = tempfile.mkdtemp()
    def dummy_input_fn():
      return tf.ones([1, 2]), tf.ones([1, 64, 64, 3])

    # Mock the input function so that we do not have to load actual image data.
    # Making sure the model builds and trains is enough.
    with mock.patch.object(faces_gan, 'get_get_batch',
                           return_value=dummy_input_fn):
      faces_gan.main(None)

if __name__ == '__main__':
  tf.test.main()
