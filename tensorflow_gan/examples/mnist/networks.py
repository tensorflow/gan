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

"""Networks for MNIST example using TF-GAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_probability as tfp

ds = tfp.distributions


def _dense(inputs, units, l2_weight):
  return tf.compat.v1.layers.dense(
      inputs,
      units,
      use_bias=False,
      kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight))


def _batch_norm(inputs, is_training):
  return tf.compat.v1.layers.batch_normalization(
      inputs, momentum=0.999, epsilon=0.001, training=is_training)


def _deconv2d(inputs, filters, kernel_size, stride, l2_weight):
  return tf.compat.v1.layers.conv2d_transpose(
      inputs,
      filters,
      kernel_size,
      strides=stride,
      padding='same',
      kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight))


def _conv2d(inputs, filters, kernel_size, stride, l2_weight):
  return tf.compat.v1.layers.conv2d(
      inputs,
      filters,
      kernel_size,
      strides=stride,
      padding='same',
      kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight))


def generator_helper(
    noise, is_conditional, one_hot_labels, weight_decay, is_training):
  """Core MNIST generator.

  This function is reused between the different GAN modes (unconditional,
  conditional, etc).

  Args:
    noise: A 2D Tensor of shape [batch size, noise dim].
    is_conditional: Whether to condition on labels.
    one_hot_labels: Optional labels for conditioning.
    weight_decay: The value of the l2 weight decay.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A generated image in the range [-1, 1].
  """
  net = _dense(noise, 1024, weight_decay)
  net = _batch_norm(net, is_training)
  net = tf.nn.relu(net)

  if is_conditional:
    net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)

  net = _dense(net, 7 * 7 * 128, weight_decay)
  net = _batch_norm(net, is_training)
  net = tf.nn.relu(net)

  net = tf.reshape(net, [-1, 7, 7, 128])

  net = _deconv2d(net, 64, 4, 2, weight_decay)
  net = _batch_norm(net, is_training)
  net = tf.nn.relu(net)

  net = _deconv2d(net, 32, 4, 2, weight_decay)
  net = _batch_norm(net, is_training)
  net = tf.nn.relu(net)

  # Output should have 1 pixel (grayscale).
  net = _conv2d(net, 1, 4, 1, weight_decay)

  # Make sure that generator output is in the same range as `inputs`
  # ie [-1, 1].
  net = tf.tanh(net)

  return net


def unconditional_generator(noise, weight_decay=2.5e-5, is_training=True):
  """Generator to produce unconditional MNIST images.

  Args:
    noise: A single Tensor representing noise.
    weight_decay: The value of the l2 weight decay.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A generated image in the range [-1, 1].
  """
  return generator_helper(noise, False, None, weight_decay, is_training)


def conditional_generator(inputs, weight_decay=2.5e-5, is_training=True):
  """Generator to produce MNIST images conditioned on class.

  Args:
    inputs: A 2-tuple of Tensors (noise, one_hot_labels).
    weight_decay: The value of the l2 weight decay.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A generated image in the range [-1, 1].s
  """
  noise, one_hot_labels = inputs
  return generator_helper(noise, True, one_hot_labels, weight_decay,
                          is_training)


def infogan_generator(inputs, categorical_dim, weight_decay=2.5e-5,
                      is_training=True):
  """InfoGAN generator network on MNIST digits.

  Based on a paper https://arxiv.org/abs/1606.03657, their code
  https://github.com/openai/InfoGAN, and code by pooleb@.

  Args:
    inputs: A 3-tuple of Tensors (unstructured_noise, categorical structured
      noise, continuous structured noise). `inputs[0]` and `inputs[2]` must be
      2D, and `inputs[1]` must be 1D. All must have the same first dimension.
    categorical_dim: Dimensions of the incompressible categorical noise.
    weight_decay: The value of the l2 weight decay.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A generated image in the range [-1, 1].
  """
  unstructured_noise, cat_noise, cont_noise = inputs
  cat_noise_onehot = tf.one_hot(cat_noise, categorical_dim)
  all_noise = tf.concat(
      [unstructured_noise, cat_noise_onehot, cont_noise], axis=1)
  return generator_helper(all_noise, False, None, weight_decay, is_training)


leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)


def discriminator_helper(img, is_conditional, one_hot_labels, weight_decay):
  """Core MNIST discriminator.

  This function is reused between the different GAN modes (unconditional,
  conditional, etc).

  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    is_conditional: Whether to condition on labels.
    one_hot_labels: Labels to optionally condition the network on.
    weight_decay: The L2 weight decay.

  Returns:
    Final fully connected discriminator layer. [batch_size, 1024].
  """
  sn_gettr = tfgan.features.spectral_normalization_custom_getter
  with tf.compat.v1.variable_scope('sn', custom_getter=sn_gettr(training=True)):
    net = _conv2d(img, 64, 4, 2, weight_decay)
    net = leaky_relu(net)

    net = _conv2d(net, 128, 4, 2, weight_decay)
    net = leaky_relu(net)

    net = tf.compat.v1.layers.flatten(net)

    if is_conditional:
      net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)

    net = _dense(net, 1024, weight_decay)
    net = _batch_norm(net, is_training=True)
    net = leaky_relu(net)

  return net


def unconditional_discriminator(img, unused_conditioning, weight_decay=2.5e-5):
  """Discriminator network on unconditional MNIST digits.

  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    unused_conditioning: The TFGAN API can help with conditional GANs, which
      would require extra `condition` information to both the generator and the
      discriminator. Since this example is not conditional, we do not use this
      argument.
    weight_decay: The L2 weight decay.

  Returns:
    Logits for the probability that the image is real.
  """
  net = discriminator_helper(img, False, None, weight_decay)
  return tf.compat.v1.layers.dense(
      net, 1, kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))


def conditional_discriminator(img, conditioning, weight_decay=2.5e-5):
  """Conditional discriminator network on MNIST digits.

  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    conditioning: A 2-tuple of Tensors representing (noise, one_hot_labels).
    weight_decay: The L2 weight decay.

  Returns:
    Logits for the probability that the image is real.
  """
  _, one_hot_labels = conditioning
  net = discriminator_helper(img, True, one_hot_labels, weight_decay)
  return tf.compat.v1.layers.dense(
      net, 1, kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))


def infogan_discriminator(img, unused_conditioning, weight_decay=2.5e-5,
                          categorical_dim=10, continuous_dim=2):
  """InfoGAN discriminator network on MNIST digits.

  Based on a paper https://arxiv.org/abs/1606.03657, their code
  https://github.com/openai/InfoGAN, and code by pooleb@.

  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    unused_conditioning: The TFGAN API can help with conditional GANs, which
      would require extra `condition` information to both the generator and the
      discriminator. Since this example is not conditional, we do not use this
      argument.
    weight_decay: The L2 weight decay.
    categorical_dim: Dimensions of the incompressible categorical noise.
    continuous_dim: Dimensions of the incompressible continuous noise.

  Returns:
    Logits for the probability that the image is real, and a list of posterior
    distributions for each of the noise vectors.
  """
  net = discriminator_helper(img, False, None, weight_decay)
  logits_real = tf.compat.v1.layers.dense(net, 1)

  # Compute logits for each category of categorical latent.
  logits_cat = tf.compat.v1.layers.dense(net, categorical_dim)
  q_cat = ds.Categorical(logits_cat)

  # Compute mean for Gaussian posterior of continuous latents.
  mu_cont = tf.compat.v1.layers.dense(net, continuous_dim)
  sigma_cont = tf.ones_like(mu_cont)
  q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)

  return logits_real, [q_cat, q_cont]
