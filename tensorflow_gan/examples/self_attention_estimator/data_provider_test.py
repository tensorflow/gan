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

"""Tests for self_attention_estimator.data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import PIL
import tensorflow as tf
from tensorflow_gan.examples.self_attention_estimator import data_provider

mock = tf.compat.v1.test.mock


# Following two functions were copied from the image transformation routines in
# https://github.com/openai/improved-gan/blob/master/imagenet/convert_imagenet_to_records.py
def _center_crop(x, crop_h, crop_w=None, resize_w=64):
  h, w = x.shape[:2]
  # we changed this to override the original DCGAN-TensorFlow behavior
  # Just use as much of the image as possible while keeping it square
  crop_h = min(h, w)

  if crop_w is None:
    crop_w = crop_h
  j = int(round((h - crop_h) / 2.))
  i = int(round((w - crop_w) / 2.))
  # Original code uses a deprecated method:
  # return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
  #                            [resize_w, resize_w])
  return np.array(
      PIL.Image.fromarray(x[j:j + crop_h, i:i + crop_w]).resize(
          [resize_w, resize_w], PIL.Image.BILINEAR))


def _transform(image, npx=64, is_crop=True, resize_w=64):
  del is_crop
  # npx : # of pixels width/height of image
  cropped_image = _center_crop(image, npx, resize_w=resize_w)
  return np.array(cropped_image) / 127.5 - 1.




class DataProviderTest(tf.test.TestCase, parameterized.TestCase):

  @mock.patch.object(data_provider, '_load_imagenet_dataset', autospec=True)
  def test_provide_data_shape(self, mock_ds):
    batch_size = 16
    num_batches = 3
    mock_ds.return_value = tf.data.Dataset.from_tensors(
        np.zeros([128, 128, 3])).map(lambda x: {'image': x, 'label': 1})
    batches = data_provider.provide_data(
        batch_size=batch_size,
        num_batches=num_batches,
        shuffle_buffer_size=10)
    self.assertLen(batches, num_batches)
    for img, lbl in batches:
      img.shape.assert_is_compatible_with([batch_size, 128, 128, 3])
      lbl.shape.assert_is_compatible_with([batch_size, 1])

  def test_preprocess_dataset_record_shapes(self):
    dummy_record = {
        'image': tf.zeros([123, 456, 3], dtype=tf.uint8),
        'label': tf.constant([4]),
    }
    process_fn = data_provider._preprocess_dataset_record_fn(image_size=128)
    processed_record = process_fn(dummy_record)
    processed_record[0].shape.assert_is_compatible_with([128, 128, 3])
    processed_record[1].shape.assert_is_compatible_with([1])

  def test_preprocess_dataset_record_centering(self):
    """Checks that `_preprocess_dataset_record` correctly crops image."""
    center_size = 4
    padding_size = 5
    dummy_record = {
        'image':
            tf.concat([
                tf.zeros([center_size, padding_size, 3], dtype=tf.uint8),
                255 * tf.ones([center_size, center_size, 3], dtype=tf.uint8),
                tf.zeros([center_size, padding_size, 3], dtype=tf.uint8)
            ],
                      axis=1),
        'label':
            tf.constant([4]),
    }
    image_size = 7
    process_fn = data_provider._preprocess_dataset_record_fn(
        image_size=image_size)
    processed_record = process_fn(dummy_record)
    processed_record[0].shape.assert_is_compatible_with(
        [image_size, image_size, 3])
    # Test that output is all-ones (ignore the boundary in the check).
    self.assertAllEqual(processed_record[0][1:-1, 1:-1, :],
                        tf.ones([image_size - 2, image_size - 2, 3]))

  @parameterized.parameters(
      {'nrows': 128, 'ncols': 128},
      {'nrows': 234, 'ncols': 100},
      {'nrows': 100, 'ncols': 234},
  )
  def test_compare_preprocess_with_improved_gan(self, nrows, ncols):
    """Compares the image resizing function with that of openai/improved-gan."""
    if tf.executing_eagerly():
      # Eval is not supported when eager execution is enabled.
      return
    test_image = []
    for j in range(nrows):
      test_image.append([[(i // 2 + j) % 256] * 3 for i in range(ncols)])
    test_image = np.array(test_image, dtype=np.uint8)
    improved_image = _transform(test_image, npx=128, is_crop=True, resize_w=128)
    dummy_record = {
        'image': tf.constant(test_image, dtype=tf.uint8),
        'label': [4],
    }
    process_fn = data_provider._preprocess_dataset_record_fn(image_size=128)
    processed_record = process_fn(dummy_record)
    with self.session():
      # There's a relatively large gap between the two results, mainly because
      # of PIL's sampling strategy. E.g.,
      # """
      # test_image = np.array([[0], [255]], dtype=np.uint8)
      # PIL.Image.fromarray(test_image).resize([2, 2], PIL.Image.BILINEAR)
      # """
      # results with [[0, 0], [127, 127]] (instead of [[0, 0], [255, 255]]).
      self.assertLess(
          tf.norm(tensor=improved_image - processed_record[0],
                  ord=np.inf).eval(), 4. / 256.)


if __name__ == '__main__':
  tf.test.main()
