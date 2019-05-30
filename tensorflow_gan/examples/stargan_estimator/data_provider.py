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

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_gan.examples.cyclegan import data_provider as cyclegan_dp
from tensorflow_gan.examples.stargan import data_provider


provide_data = data_provider.provide_data


def provide_celeba_test_set(patch_size):
  """Provide one example of every class.

  Args:
    patch_size: Python int. The patch size to extract.

  Returns:
    An `np.array` of shape (num_domains, H, W, C) representing the images.
      Values are in [-1, 1].
  """
  ds = tfds.load('celeb_a', split='test')
  def _preprocess(x):
    return {
        'image': cyclegan_dp.full_image_to_patch(x['image'], patch_size),
        'attributes': x['attributes'],
    }
  ds = ds.map(_preprocess)
  ds_np = tfds.as_numpy(ds)

  # Get one image of each hair type.
  images = []
  labels = []
  while len(images) < 3:
    elem = next(ds_np)
    attr = elem['attributes']
    cur_lbl = [attr['Black_Hair'], attr['Blond_Hair'], attr['Brown_Hair']]
    if cur_lbl not in labels:
      images.append(elem['image'])
      labels.append(cur_lbl)
  images = np.array(images, dtype=np.float32)

  assert images.dtype == np.float32
  assert np.max(np.abs(images)) <= 1.0
  assert images.shape == (3, patch_size, patch_size, 3)

  return images
