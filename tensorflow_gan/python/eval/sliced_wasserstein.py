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

"""Implementation of Sliced Wasserstein Distance.

Proposed in https://arxiv.org/abs/1710.10196 and the official Theano
implementation that we used as reference can be found here:
https://github.com/tkarras/progressive_growing_of_gans

Note: this is not an exact distance but an approximation through random
projections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_gan.python import contrib_utils as contrib

__all__ = ['sliced_wasserstein_distance']

_GAUSSIAN_FILTER = np.float32([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]]).reshape([5, 5, 1, 1]) / 256.0


def _to_float(tensor):
  return tf.cast(tensor, tf.float32)


def laplacian_pyramid(batch, num_levels):
  """Compute a Laplacian pyramid.

  Args:
      batch: (tensor) The batch of images (batch, height, width, channels).
      num_levels: (int) Desired number of hierarchical levels.

  Returns:
      List of tensors from the highest to lowest resolution.
  """
  gaussian_filter = tf.constant(_GAUSSIAN_FILTER)

  def spatial_conv(batch, gain):
    """Custom conv2d."""
    s = tf.shape(input=batch)
    padded = tf.pad(
        tensor=batch, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
    xt = tf.transpose(a=padded, perm=[0, 3, 1, 2])
    xt = tf.reshape(xt, [s[0] * s[3], s[1] + 4, s[2] + 4, 1])
    conv_out = tf.nn.conv2d(
        input=xt,
        filters=gaussian_filter * gain,
        strides=[1] * 4,
        padding='VALID')
    conv_xt = tf.reshape(conv_out, [s[0], s[3], s[1], s[2]])
    conv_xt = tf.transpose(a=conv_xt, perm=[0, 2, 3, 1])
    return conv_xt

  def pyr_down(batch):  # matches cv2.pyrDown()
    return spatial_conv(batch, 1)[:, ::2, ::2]

  def pyr_up(batch):  # matches cv2.pyrUp()
    s = tf.shape(input=batch)
    zeros = tf.zeros([3 * s[0], s[1], s[2], s[3]])
    res = tf.concat([batch, zeros], 0)
    res = contrib.batch_to_space(
        input=res, crops=[[0, 0], [0, 0]], block_shape=2)
    res = spatial_conv(res, 4)
    return res

  pyramid = [_to_float(batch)]
  for _ in range(1, num_levels):
    pyramid.append(pyr_down(pyramid[-1]))
    pyramid[-2] -= pyr_up(pyramid[-1])
  return pyramid


def _batch_to_patches(batch, patches_per_image, patch_size):
  """Extract patches from a batch.

  Args:
      batch: (tensor) The batch of images (batch, height, width, channels).
      patches_per_image: (int) Number of patches to extract per image.
      patch_size: (int) Size of the patches (size, size, channels) to extract.

  Returns:
      Tensor (batch*patches_per_image, patch_size, patch_size, channels) of
      patches.
  """

  def py_func_random_patches(batch):
    """Numpy wrapper."""
    batch_size, height, width, channels = batch.shape
    patch_count = patches_per_image * batch_size
    hs = patch_size // 2
    # Randomly pick patches.
    patch_id, y, x, chan = np.ogrid[0:patch_count, -hs:hs + 1, -hs:hs + 1, 0:3]
    img_id = patch_id // patches_per_image
    # pylint: disable=g-no-augmented-assignment
    # Need explicit addition for broadcast to work properly.
    y = y + np.random.randint(hs, height - hs, size=(patch_count, 1, 1, 1))
    x = x + np.random.randint(hs, width - hs, size=(patch_count, 1, 1, 1))
    # pylint: enable=g-no-augmented-assignment
    idx = ((img_id * height + y) * width + x) * channels + chan
    patches = batch.flat[idx]
    return patches

  patches = tf.compat.v1.py_func(
      py_func_random_patches, [batch], batch.dtype, stateful=False)
  return patches


def _normalize_patches(patches):
  """Normalize patches by their mean and standard deviation.

  Args:
      patches: (tensor) The batch of patches (batch, size, size, channels).

  Returns:
      Tensor (batch, size, size, channels) of the normalized patches.
  """
  patches = tf.concat(patches, 0)
  mean, variance = tf.nn.moments(x=patches, axes=[1, 2, 3], keepdims=True)
  patches = (patches - mean) / tf.sqrt(variance)
  return tf.reshape(patches, [tf.shape(input=patches)[0], -1])


def _sort_rows(matrix, num_rows):
  """Sort matrix rows by the last column.

  Args:
      matrix: a matrix of values (row,col).
      num_rows: (int) number of sorted rows to return from the matrix.

  Returns:
      Tensor (num_rows, col) of the sorted matrix top K rows.
  """
  tmatrix = tf.transpose(a=matrix, perm=[1, 0])
  sorted_tmatrix = tf.nn.top_k(tmatrix, num_rows)[0]
  return tf.transpose(a=sorted_tmatrix, perm=[1, 0])


def _sliced_wasserstein(a, b, random_sampling_count, random_projection_dim):
  """Compute the approximate sliced Wasserstein distance.

  Args:
      a: (matrix) Distribution "a" of samples (row, col).
      b: (matrix) Distribution "b" of samples (row, col).
      random_sampling_count: (int) Number of random projections to average.
      random_projection_dim: (int) Dimension of the random projection space.

  Returns:
      Float containing the approximate distance between "a" and "b".
  """
  s = tf.shape(input=a)
  means = []
  for _ in range(random_sampling_count):
    # Random projection matrix.
    proj = tf.random.normal([tf.shape(input=a)[1], random_projection_dim])
    proj *= tf.math.rsqrt(
        tf.reduce_sum(input_tensor=tf.square(proj), axis=0, keepdims=True))
    # Project both distributions and sort them.
    proj_a = tf.matmul(a, proj)
    proj_b = tf.matmul(b, proj)
    proj_a = _sort_rows(proj_a, s[0])
    proj_b = _sort_rows(proj_b, s[0])
    # Pairwise Wasserstein distance.
    wdist = tf.reduce_mean(input_tensor=tf.abs(proj_a - proj_b))
    means.append(wdist)
  return tf.reduce_mean(input_tensor=means)


def _sliced_wasserstein_svd(a, b):
  """Compute the approximate sliced Wasserstein distance using an SVD.

  This is not part of the paper, it's a variant with possibly more accurate
  measure.

  Args:
      a: (matrix) Distribution "a" of samples (row, col).
      b: (matrix) Distribution "b" of samples (row, col).

  Returns:
      Float containing the approximate distance between "a" and "b".
  """
  s = tf.shape(input=a)
  # Random projection matrix.
  sig, u = tf.linalg.svd(tf.concat([a, b], 0))[:2]
  proj_a, proj_b = tf.split(u * sig, 2, axis=0)
  proj_a = _sort_rows(proj_a[:, ::-1], s[0])
  proj_b = _sort_rows(proj_b[:, ::-1], s[0])
  # Pairwise Wasserstein distance.
  wdist = tf.reduce_mean(input_tensor=tf.abs(proj_a - proj_b))
  return wdist


def sliced_wasserstein_distance(real_images,
                                fake_images,
                                resolution_min=16,
                                patches_per_image=64,
                                patch_size=7,
                                random_sampling_count=1,
                                random_projection_dim=7 * 7 * 3,
                                use_svd=False):
  """Compute the Wasserstein distance between two distributions of images.

  Note that measure vary with the number of images. Use 8192 images to get
  numbers comparable to the ones in the original paper.

  Args:
      real_images: (tensor) Real images (batch, height, width, channels).
      fake_images: (tensor) Fake images (batch, height, width, channels).
      resolution_min: (int) Minimum resolution for the Laplacian pyramid.
      patches_per_image: (int) Number of patches to extract per image per
        Laplacian level.
      patch_size: (int) Width of a square patch.
      random_sampling_count: (int) Number of random projections to average.
      random_projection_dim: (int) Dimension of the random projection space.
      use_svd: experimental method to compute a more accurate distance.

  Returns:
      List of tuples (distance_real, distance_fake) for each level of the
      Laplacian pyramid from the highest resolution to the lowest.
        distance_real is the Wasserstein distance between real images
        distance_fake is the Wasserstein distance between real and fake images.
  Raises:
      ValueError: If the inputs shapes are incorrect. Input tensor dimensions
      (batch, height, width, channels) are expected to be known at graph
      construction time. In addition height and width must be the same and the
      number of colors should be exactly 3. Real and fake images must have the
      same size.
  """
  height = real_images.shape[1]
  real_images.shape.assert_is_compatible_with([None, None, height, 3])
  fake_images.shape.assert_is_compatible_with(real_images.shape)

  # Select resolutions.
  resolution_full = int(height)
  resolution_min = min(resolution_min, resolution_full)
  resolution_max = resolution_full
  # Base loss of detail.
  resolutions = [
      2**i for i in range(
          int(np.log2(resolution_max)),
          int(np.log2(resolution_min)) - 1, -1)
  ]

  # Gather patches for each level of the Laplacian pyramids.
  patches_real, patches_fake, patches_test = (
      [[] for _ in resolutions] for _ in range(3))
  for lod, level in enumerate(laplacian_pyramid(real_images, len(resolutions))):
    patches_real[lod].append(
        _batch_to_patches(level, patches_per_image, patch_size))
    patches_test[lod].append(
        _batch_to_patches(level, patches_per_image, patch_size))

  for lod, level in enumerate(laplacian_pyramid(fake_images, len(resolutions))):
    patches_fake[lod].append(
        _batch_to_patches(level, patches_per_image, patch_size))

  for lod in range(len(resolutions)):
    for patches in [patches_real, patches_test, patches_fake]:
      patches[lod] = _normalize_patches(patches[lod])

  # Evaluate scores.
  scores = []
  for lod in range(len(resolutions)):
    if not use_svd:
      scores.append(
          (_sliced_wasserstein(patches_real[lod], patches_test[lod],
                               random_sampling_count, random_projection_dim),
           _sliced_wasserstein(patches_real[lod], patches_fake[lod],
                               random_sampling_count, random_projection_dim)))
    else:
      scores.append(
          (_sliced_wasserstein_svd(patches_real[lod], patches_test[lod]),
           _sliced_wasserstein_svd(patches_real[lod], patches_fake[lod])))
  return scores
