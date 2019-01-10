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
# ============================================================================
"""StarGAN data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_gan.examples.cyclegan import data_provider


def provide_data(image_file_patterns, batch_size, patch_size):
  """Data provider wrapper on for the data_provider in gan/cyclegan.

  Args:
    image_file_patterns: A list of file pattern globs.
    batch_size: Python int. Batch size.
    patch_size: Python int. The patch size to extract.

  Returns:
    List of `Tensor` of shape (N, H, W, C) representing the images.
    List of `Tensor` of shape (N, num_domains) representing the labels.
  """

  images = data_provider.provide_custom_data(
      image_file_patterns,
      batch_size=batch_size,
      patch_size=patch_size)

  num_domains = len(images)
  labels = [tf.one_hot([idx] * batch_size, num_domains) for idx in
            range(num_domains)]

  return images, labels
