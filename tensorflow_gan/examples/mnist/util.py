# coding=utf-8
# Copyright 2020 The TensorFlow GAN Authors.
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
"""A utility for evaluating MNIST generative models.

These functions use a pretrained MNIST classifier with ~99% eval accuracy to
measure various aspects of the quality of generated MNIST digits.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub
import tensorflow_probability as tfp

ds = tfp.distributions

__all__ = [
    'mnist_score',
    'mnist_frechet_distance',
    'mnist_cross_entropy',
    'get_eval_noise_categorical',
    'get_eval_noise_continuous_dim1',
    'get_eval_noise_continuous_dim2',
    'get_infogan_noise',
]

MNIST_MODULE = 'https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1'


def mnist_score(images, num_batches=1):
    """Get MNIST classifier score.

  Args:
    images: A minibatch tensor of MNIST digits. Shape must be [batch, 28, 28,
      1].
    num_batches: Number of batches to split `generated_images` in to in order to
      efficiently run them through Inception.

  Returns:
    The classifier score, a floating-point scalar.
  """
    images.shape.assert_is_compatible_with([None, 28, 28, 1])
    mnist_classifier_fn = tfhub.load(MNIST_MODULE)
    score = tfgan.eval.classifier_score(images, mnist_classifier_fn,
                                        num_batches)
    score.shape.assert_is_compatible_with([])

    return score


def mnist_frechet_distance(real_images, generated_images, num_batches=1):
    """Frechet distance between real and generated images.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Please see TF-GAN for implementation details.

  Args:
    real_images: Real images to use to compute Frechet Inception distance.
    generated_images: Generated images to use to compute Frechet Inception
      distance.
    num_batches: Number of batches to split images into in order to efficiently
      run them through the classifier network.

  Returns:
    The Frechet distance. A floating-point scalar.
  """
    real_images.shape.assert_is_compatible_with([None, 28, 28, 1])
    generated_images.shape.assert_is_compatible_with([None, 28, 28, 1])
    mnist_classifier_fn = tfhub.load(MNIST_MODULE)
    frechet_distance = tfgan.eval.frechet_classifier_distance(
        real_images, generated_images, mnist_classifier_fn, num_batches)
    frechet_distance.shape.assert_is_compatible_with([])

    return frechet_distance


def mnist_cross_entropy(images, one_hot_labels):
    """Returns the cross entropy loss of the classifier on images.

  Args:
    images: A minibatch tensor of MNIST digits. Shape must be [batch, 28, 28,
      1].
    one_hot_labels: The one hot label of the examples. Tensor size is [batch,
      10].

  Returns:
    A scalar Tensor representing the cross entropy of the image minibatch.
  """
    logits = tfhub.load(MNIST_MODULE)(images)
    return tf.compat.v1.losses.softmax_cross_entropy(one_hot_labels,
                                                     logits,
                                                     loss_collection=None)


# TODO(joelshor): Refactor the `eval_noise` functions to reuse code.
def get_eval_noise_categorical(noise_samples, categorical_sample_points,
                               continuous_sample_points,
                               unstructured_noise_dims, continuous_noise_dims):
    """Create noise showing impact of categorical noise in InfoGAN.

  Categorical noise is constant across columns. Other noise is constant across
  rows.

  Args:
    noise_samples: Number of non-categorical noise samples to use.
    categorical_sample_points: Possible categorical noise points to sample.
    continuous_sample_points: Possible continuous noise points to sample.
    unstructured_noise_dims: Dimensions of the unstructured noise.
    continuous_noise_dims: Dimensions of the continuous noise.

  Returns:
    Unstructured noise, categorical noise, continuous noise numpy arrays. Each
    should have shape [noise_samples, ?].
  """
    rows, cols = noise_samples, len(categorical_sample_points)

    # Take random draws for non-categorical noise, making sure they are constant
    # across columns.
    unstructured_noise = []
    for _ in xrange(rows):
        cur_sample = np.random.normal(size=[1, unstructured_noise_dims])
        unstructured_noise.extend([cur_sample] * cols)
    unstructured_noise = np.concatenate(unstructured_noise)

    continuous_noise = []
    for _ in xrange(rows):
        cur_sample = np.random.choice(continuous_sample_points,
                                      size=[1, continuous_noise_dims])
        continuous_noise.extend([cur_sample] * cols)
    continuous_noise = np.concatenate(continuous_noise)

    # Increase categorical noise from left to right, making sure they are constant
    # across rows.
    categorical_noise = np.tile(categorical_sample_points, rows)

    return unstructured_noise, categorical_noise, continuous_noise


def get_eval_noise_continuous_dim1(noise_samples, categorical_sample_points,
                                   continuous_sample_points,
                                   unstructured_noise_dims,
                                   continuous_noise_dims):  # pylint:disable=unused-argument
    """Create noise showing impact of first dim continuous noise in InfoGAN.

  First dimension of continuous noise is constant across columns. Other noise is
  constant across rows.

  Args:
    noise_samples: Number of non-categorical noise samples to use.
    categorical_sample_points: Possible categorical noise points to sample.
    continuous_sample_points: Possible continuous noise points to sample.
    unstructured_noise_dims: Dimensions of the unstructured noise.
    continuous_noise_dims: Dimensions of the continuous noise.

  Returns:
    Unstructured noise, categorical noise, continuous noise numpy arrays.
  """
    rows, cols = noise_samples, len(continuous_sample_points)

    # Take random draws for non-first-dim-continuous noise, making sure they are
    # constant across columns.
    unstructured_noise = []
    for _ in xrange(rows):
        cur_sample = np.random.normal(size=[1, unstructured_noise_dims])
        unstructured_noise.extend([cur_sample] * cols)
    unstructured_noise = np.concatenate(unstructured_noise)

    categorical_noise = []
    for _ in xrange(rows):
        cur_sample = np.random.choice(categorical_sample_points)
        categorical_noise.extend([cur_sample] * cols)
    categorical_noise = np.array(categorical_noise)

    cont_noise_dim2 = []
    for _ in xrange(rows):
        cur_sample = np.random.choice(continuous_sample_points, size=[1, 1])
        cont_noise_dim2.extend([cur_sample] * cols)
    cont_noise_dim2 = np.concatenate(cont_noise_dim2)

    # Increase first dimension of continuous noise from left to right, making sure
    # they are constant across rows.
    cont_noise_dim1 = np.expand_dims(np.tile(continuous_sample_points, rows),
                                     1)

    continuous_noise = np.concatenate((cont_noise_dim1, cont_noise_dim2), 1)

    return unstructured_noise, categorical_noise, continuous_noise


def get_eval_noise_continuous_dim2(noise_samples, categorical_sample_points,
                                   continuous_sample_points,
                                   unstructured_noise_dims,
                                   continuous_noise_dims):  # pylint:disable=unused-argument
    """Create noise showing impact of second dim of continuous noise in InfoGAN.

  Second dimension of continuous noise is constant across columns. Other noise
  is constant across rows.

  Args:
    noise_samples: Number of non-categorical noise samples to use.
    categorical_sample_points: Possible categorical noise points to sample.
    continuous_sample_points: Possible continuous noise points to sample.
    unstructured_noise_dims: Dimensions of the unstructured noise.
    continuous_noise_dims: Dimensions of the continuous noise.

  Returns:
    Unstructured noise, categorical noise, continuous noise numpy arrays.
  """
    rows, cols = noise_samples, len(continuous_sample_points)

    # Take random draws for non-first-dim-continuous noise, making sure they are
    # constant across columns.
    unstructured_noise = []
    for _ in xrange(rows):
        cur_sample = np.random.normal(size=[1, unstructured_noise_dims])
        unstructured_noise.extend([cur_sample] * cols)
    unstructured_noise = np.concatenate(unstructured_noise)

    categorical_noise = []
    for _ in xrange(rows):
        cur_sample = np.random.choice(categorical_sample_points)
        categorical_noise.extend([cur_sample] * cols)
    categorical_noise = np.array(categorical_noise)

    cont_noise_dim1 = []
    for _ in xrange(rows):
        cur_sample = np.random.choice(continuous_sample_points, size=[1, 1])
        cont_noise_dim1.extend([cur_sample] * cols)
    cont_noise_dim1 = np.concatenate(cont_noise_dim1)

    # Increase first dimension of continuous noise from left to right, making sure
    # they are constant across rows.
    cont_noise_dim2 = np.expand_dims(np.tile(continuous_sample_points, rows),
                                     1)

    continuous_noise = np.concatenate((cont_noise_dim1, cont_noise_dim2), 1)

    return unstructured_noise, categorical_noise, continuous_noise


def get_infogan_noise(batch_size, categorical_dim, structured_continuous_dim,
                      total_continuous_noise_dims):
    """Get unstructured and structured noise for InfoGAN.

  Args:
    batch_size: The number of noise vectors to generate.
    categorical_dim: The number of categories in the categorical noise.
    structured_continuous_dim: The number of dimensions of the uniform
      continuous noise.
    total_continuous_noise_dims: The number of continuous noise dimensions. This
      number includes the structured and unstructured noise.

  Returns:
    A 2-tuple of structured and unstructured noise. First element is the
    unstructured noise, and the second is a 2-tuple of
    (categorical structured noise, continuous structured noise).
  """
    # Get unstructurd noise.
    unstructured_noise = tf.random.normal(
        [batch_size, total_continuous_noise_dims - structured_continuous_dim])

    # Get categorical noise Tensor.
    categorical_dist = ds.Categorical(logits=tf.zeros([categorical_dim]))
    categorical_noise = categorical_dist.sample([batch_size])

    # Get continuous noise Tensor.
    continuous_dist = ds.Uniform(-tf.ones([structured_continuous_dim]),
                                 tf.ones([structured_continuous_dim]))
    continuous_noise = continuous_dist.sample([batch_size])

    return [unstructured_noise], [categorical_noise, continuous_noise]
