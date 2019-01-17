# coding=utf-8
# Copyright 2018 The TensorFlow GAN Authors.
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

"""Testing faces inpainter.

Tests for g3.tp.tensorflow_models.gan.face_inpainting.faces_inpainter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import flagsaver
import mock
import numpy as np
import tensorflow as tf
from tensorflow_gan.examples.face_inpainting import faces_inpainter

FLAGS = flags.FLAGS


class FacesInpainterTest(tf.test.TestCase):

  def setUp(self):
    """Set up mocks used in both tests."""
    super(FacesInpainterTest, self).setUp()

    # Set up mock variable outputs. Imshape here represents the full image,
    # including channels.
    self.imshape = (64, 64, 3)
    self.dummy_image = np.random.random_sample(self.imshape).astype(np.float32)

    # Mock the image loading so we do not have to actually load image dirs.
    self.mock_open_image = mock.patch.object(
        faces_inpainter.faces_gan, 'open_image', return_value=self.dummy_image)

  # Mock the random mask generation so that the input_fn is consistent.
  @mock.patch.object(faces_inpainter.random, 'randint', return_value=10)
  def test_get_input_fn(self, mock_randint):
    """Make sure input is wired correctly, applies weight mask correctly."""
    del mock_randint
    mask_shape = self.imshape[0:2]
    true_mask = np.ones(mask_shape)
    true_mask[10:20, 10:20] = 0
    dummy_files = tf.data.Dataset.from_tensors(
        tf.constant(['dummy', 'dummy'], dtype=tf.string))

    # Mock list_files to avoid issues with the expected batch size.
    with mock.patch.object(faces_inpainter.tf.data.Dataset,
                           'list_files', return_value=dummy_files):
      with self.mock_open_image:
        input_fn = faces_inpainter.get_input_fn('dummy', 2, mask_shape, 10)
        features, labels = input_fn()

    sess = tf.Session()
    output_images = sess.run(labels)
    output_mask = sess.run(features['mask'])
    weighted_mask = sess.run(features['weighted_mask'])

    # Make sure the image gets passed through as expected, and check that the
    # generated 'random' mask has the correct blacked out component.
    self.assertTrue(np.array_equal(
        output_images, [self.dummy_image, self.dummy_image]))
    self.assertTrue(np.array_equal(output_mask, np.expand_dims(true_mask, -1)))

    # To check the validity of the weighted mask, make sure that pixels sampled
    # closer to the masked area have higher weight than those sampled farther.
    for i in range(0, 10):
      self.assertGreaterEqual(weighted_mask[21 + i, 21 + i],
                              weighted_mask[20 + i, 20 + i])

  @flagsaver.flagsaver
  def test_inpainter_model(self):
    """Make sure the inpainter model builds and can train."""

    # Set up variables for the mocked test.
    FLAGS.num_steps = 2
    FLAGS.batch_size = 2
    FLAGS.add_summaries = False
    FLAGS.warmstart = False
    tempdir = self.create_tempdir()
    FLAGS.save_dir = tempdir.full_path

    # Create two named files to make sure list_dir works as intended. list_dir
    # is difficult to mock in this situation because tensorflow requires the
    # dataset created from list_dir to be from the same underlying graph. These
    # are automatically cleaned up at the end of the test.
    tempdir.create_file('file_one')
    tempdir.create_file('file_two')

    with self.mock_open_image:
      faces_inpainter.main(None)


if __name__ == '__main__':
  tf.test.main()
