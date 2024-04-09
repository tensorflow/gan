# coding=utf-8
# Copyright 2024 The TensorFlow GAN Authors.
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

"""Code for training the ESRGAN model."""

import os
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_gan as tfgan


def preprocess_input(image):
  """Input images are preprocessed by subtracting the dataset mean.

  Since the VGG-19 network used here to extract features, is pretrained
  on ImageNet dataset, its mean value (103.939, 116.779, 123.68) is subtracted
  from the input images.
  This ensures that each feature will have a similar range and the gradients
  calculated during training are stable.

  Args:
    image : Batch of tensors representing an image.

  Returns:
    preprocessed images after subtracting the mean value.
  """
  image = image[..., ::-1]
  mean = -tf.constant([103.939, 116.779, 123.68])
  return tf.nn.bias_add(image, mean)


# Utility functions for training
def visualize_results(image_lr,
                      generated,
                      image_hr,
                      size=256,
                      image_dir='',
                      step=0,
                      train=True):
  """Creates an image grid using Tf-GAN's image grid function.

  Args:
      image_lr : Batch of tensors representing LR images.
      generated : Batch of tensors representing generated images.
      image_hr : Batch of tensors representing HR images.
      size : Size to which all images are resized for visualization.
      image_dir : Directory to save the results.
      step : Number of steps completed, for naming purposes.
      train : Training or Validation.
  """
  resized_lr = tf.image.resize(
      image_lr, [size, size], method=tf.image.ResizeMethod.BILINEAR)
  resized_gen = tf.image.resize(
      generated, [size, size], method=tf.image.ResizeMethod.BILINEAR)
  resized_hr = tf.image.resize(
      image_hr, [size, size], method=tf.image.ResizeMethod.BILINEAR)

  # Stack images in the order LR, Generated and HR for display
  stack = tf.stack([resized_lr[0], resized_gen[0], resized_hr[0]])

  # Generate an image grid using tf-gan's image grid function.
  image_grid = tfgan.eval.python_image_grid(stack, grid_shape=(1, 3))
  result = PIL.Image.fromarray(image_grid.astype(np.uint8))

  if image_dir:
    if train:
      os.makedirs(image_dir + 'training_results', exist_ok=True)
      result.save(image_dir + 'training_results/' + 'step_{}.png'.format(step))
    else:
      os.makedirs(image_dir + 'val_results', exist_ok=True)
      result.save(image_dir + 'val_results/' + 'step_{}.png'.format(step))


def network_interpolation(alpha=0.2, phase_1_path=None, phase_2_path=None):
  """Network interpolation as explained in section 3.4 in the paper.

  Args:
      alpha : interpolation parameter.
      phase_1_path : path to the network saved after phase 1 training.
      phase_2_path : path to the network saved after phase 2 training.

  Returns:
      Interpolated generator network.
  """
  if phase_1_path is None or phase_2_path is None:
    raise ValueError(
        'Please specify paths to both phase 1 and phase 2 generators.')

  psnr_gen = tf.keras.models.load_model(phase_1_path)
  gan_gen = tf.keras.models.load_model(phase_2_path)

  for var_1, var_2 in zip(gan_gen.trainable_variables,
                          psnr_gen.trainable_variables):
    var_1.assign((1 - alpha) * var_2 + alpha * var_1)

  return gan_gen


# Utility functions for evaluation
def get_frechet_inception_distance(real_images, generated_images, batch_size,
                                   num_inception_images):
  """Get Frechet Inception Distance between real and generated images.

  Args:
    real_images: Real images minibatch.
    generated_images: Generated images minibatch.
    batch_size: Batch dimension.
    num_inception_images: Number of images to run through Inception at once.

  Returns:
    Frechet Inception distance. A floating-point scalar.

  Raises:
    ValueError: If the minibatch size is known at graph construction time, and
      doesn't batch `batch_size`.
  """
  # Validate input dimensions.
  real_images.shape[0:1].assert_is_compatible_with([batch_size])
  generated_images.shape[0:1].assert_is_compatible_with([batch_size])

  # Resize input images.
  size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
  resized_real_images = tf.image.resize(
      real_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)
  resized_generated_images = tf.image.resize(
      generated_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)

  # Compute Frechet Inception Distance.
  num_batches = batch_size // num_inception_images
  fid = tfgan.eval.frechet_inception_distance(
      resized_real_images, resized_generated_images, num_batches=num_batches)

  return fid


def get_inception_scores(images, batch_size, num_inception_images):
  """Calculate Inception score for images.

  Args:
      images : A batch of tensors representing series of images.
      batch_size : Integer representing batch dimension.
      num_inception_images : Number of images to run through Inception at once.

  Returns:
      Inception scores : A tensor consisting of scores for each batch of images.

  Raises:
      ValueError: If `batch_size` is incompatible with the first dimension of
      `images`
      ValueError: If `batch_size` isn't divisible by `num_inception_images`.
  """

  # Validate inputs.
  images.shape[0:1].assert_is_compatible_with([batch_size])
  if batch_size % num_inception_images != 0:
    raise ValueError(
        '`batch_size` must be divisible by `num_inception_images`.')

  # Resize images.
  size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
  resized_images = tf.image.resize(
      images, [size, size], method=tf.image.ResizeMethod.BILINEAR)

  # Run images through Inception.
  num_batches = batch_size // num_inception_images

  inc_score = tfgan.eval.inception_score(
      resized_images, num_batches=num_batches)

  return inc_score


def get_psnr(real, generated):
  """Calculate PSNR values for the given samples of images.

  Args:
      real: batch of tensors representing real images.
      generated : batch of tensors representing generated images.

  Returns:
      PSNR value for the given batch of real and generated images.
  """
  psnr_value = tf.reduce_mean(tf.image.psnr(generated, real, max_val=256.0))

  return psnr_value
