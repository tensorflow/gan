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

"""StarGAN Estimator data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow_gan.examples.stargan import data_provider


provide_data = data_provider.provide_data


def provide_celeba_test_set():
  """Provide one example of every class, and labels.

  Returns:
    An `np.array` of shape (num_domains, H, W, C) representing the images.
      Values are in [-1, 1].
    An `np.array` of shape (num_domains, num_domains) representing the labels.

  Raises:
    ValueError: If test data is inconsistent or malformed.
  """
  base_dir = 'tensorflow_gan/examples/stargan_estimator/data'
  images_fn = os.path.join(base_dir, 'celeba_test_split_images.npy')
  with tf.gfile.Open(images_fn, 'rb') as f:
    images_np = np.load(f)
  labels_fn = os.path.join(base_dir, 'celeba_test_split_labels.npy')
  with tf.gfile.Open(labels_fn, 'rb') as f:
    labels_np = np.load(f)
  if images_np.shape[0] != labels_np.shape[0]:
    raise ValueError('Test data is malformed.')

  return images_np, labels_np
