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

"""Spectral normalization ops."""

import contextlib
import tensorflow as tf
import tensorflow_gan as tfgan


sn_gettr = tfgan.features.spectral_normalization_custom_getter


def snconv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, training=True,
             name='snconv2d'):
  """Creates a 2d conv-layer with Spectral Norm applied to the weights.

  Args:
    input_: 4D input tensor (batch size, height, width, channel).
    output_dim: Number of features in the output layer.
    k_h: The height of the convolutional kernel.
    k_w: The width of the convolutional kernel.
    d_h: The height stride of the convolutional kernel.
    d_w: The width stride of the convolutional kernel.
    training: If `True`, add the spectral norm assign ops.
    name: The name of the variable scope.
  Returns:
    conv: The normalized tensor.

  """
  with tf.compat.v1.variable_scope(
      name,
      custom_getter=sn_gettr(training=training, equality_constrained=False)):
    return tf.compat.v1.layers.conv2d(
        input_,
        filters=output_dim,
        kernel_size=(k_h, k_w),
        strides=(d_h, d_w),
        padding='same',
        activation=None,
        use_bias=True,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform'),
        bias_initializer=tf.compat.v1.initializers.zeros(),
        name=name)


def snlinear(x, output_size, bias_start=0.0, training=True, name='snlinear'):
  """Creates a linear layer with Spectral Normalization applied.

  Args:
    x: 2D input tensor (batch size, features).
    output_size: Integer number of features in output of layer.
    bias_start: Float to which bias parameters are initialized.
    training: If `True`, add the spectral norm assign ops.
    name: Optional, variable scope to put the layer's parameters into.
  Returns:
    The normalized output tensor of the linear layer.
  """
  with tf.compat.v1.variable_scope(
      name,
      custom_getter=sn_gettr(training=training, equality_constrained=False)):
    return tf.compat.v1.layers.dense(
        x,
        output_size,
        activation=None,
        use_bias=True,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform'),
        bias_initializer=tf.compat.v1.initializers.constant(bias_start))


def sn_embedding(x, number_classes, embedding_size, training=True,
                 name='snembedding'):
  """Creates an embedding lookup with Spectral Normalization applied.

  Args:
    x: 1D input tensor (batch size, ).
    number_classes: The number of classes.
    embedding_size: The length of the embeddding vector for each class.
    training: If `True`, add the spectral norm assign ops.
    name: Optional, variable scope to put the layer's parameters into
  Returns:
    The output tensor (batch size, embedding_size).
  """
  with tf.compat.v1.variable_scope(name):
    embedding_map = tf.compat.v1.get_variable(
        name='embedding_map',
        shape=[number_classes, embedding_size],
        initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform'))
    embedding_map_bar_transpose = tfgan.features.spectral_normalize(
        tf.transpose(a=embedding_map),
        training=training,
        equality_constrained=False)
    embedding_map_bar = tf.transpose(a=embedding_map_bar_transpose)
    return tf.nn.embedding_lookup(params=embedding_map_bar, ids=x)


class ConditionalBatchNorm(object):
  """Conditional Batch Normalization.

  The same as normal Batch Normalization, but there is a different (gamma, beta)
  pair for each possible category.
  For each  class, it has a specific gamma and beta as normalization variable.
  """

  # TODO(augustusodena) Merge conditional batch norm with batch norm.
  # TODO(augustusodena) Use more sophisticated FilM layer here.
  # TODO(augustusodena) Why does this need to be a class?
  def __init__(self, num_categories, name='conditional_batch_norm'):
    """Inits the object.

    This is just a setter.

    Args:
      num_categories: Integer number of classes (and gamma, beta pairs).
      name: String name to be used for scoping.
    Returns:
      Initialized object.
    """
    with tf.compat.v1.variable_scope(name):
      self.name = name
      self.num_categories = num_categories

  def __call__(self, inputs, labels):
    """Adds Conditional Batch norm to the TF Graph.

    Args:
      inputs: Tensor of inputs (e.g. images).
      labels: Tensor of labels - same first dimension as inputs.
    Returns:
      Output tensor.
    """
    inputs = tf.convert_to_tensor(value=inputs)
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    axis = [0, 1, 2]
    shape = tf.TensorShape([self.num_categories]).concatenate(params_shape)

    with tf.compat.v1.variable_scope(self.name):
      self.gamma = tf.compat.v1.get_variable(
          'gamma', shape, initializer=tf.compat.v1.initializers.ones())
      self.beta = tf.compat.v1.get_variable(
          'beta', shape, initializer=tf.compat.v1.initializers.zeros())
      beta = tf.gather(self.beta, labels)
      beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
      gamma = tf.gather(self.gamma, labels)
      gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)

      mean, variance = tf.nn.moments(x=inputs, axes=axis, keepdims=True)
      outputs = tf.nn.batch_normalization(
          inputs, mean, variance, beta, gamma, variance_epsilon=1e-5)
      outputs.set_shape(inputs_shape)
      return outputs


class BatchNorm(object):
  """Batch Normalization.

  This is just vanilla batch normalization.
  """

  def __init__(self, name='batch_norm'):
    """Inits the object.

    This is just a setter.

    Args:
      name: String name to be used for scoping.
    Returns:
      Initialized object.
    """
    with tf.compat.v1.variable_scope(name):
      self.name = name

  def __call__(self, inputs):
    """Adds Batch Norm to the TF Graph.

    Args:
      inputs: Tensor of inputs (e.g. images).
    Returns:
      Output tensor.
    """
    inputs = tf.convert_to_tensor(value=inputs)
    inputs_shape = inputs.get_shape().as_list()
    params_shape = inputs_shape[-1]
    axis = [0, 1, 2]
    shape = tf.TensorShape([params_shape])
    with tf.compat.v1.variable_scope(self.name):
      self.gamma = tf.compat.v1.get_variable(
          'gamma', shape, initializer=tf.compat.v1.initializers.ones())
      self.beta = tf.compat.v1.get_variable(
          'beta', shape, initializer=tf.compat.v1.initializers.zeros())
      beta = self.beta
      gamma = self.gamma

      mean, variance = tf.nn.moments(x=inputs, axes=axis, keepdims=True)
      outputs = tf.nn.batch_normalization(
          inputs, mean, variance, beta, gamma, variance_epsilon=1e-5)
      outputs.set_shape(inputs_shape)
      return outputs


def sn_conv1x1(x, output_dim, training=True, name='sn_conv1x1'):
  """Builds graph for a spectrally normalized 1 by 1 convolution.

  This is used in the context of non-local networks to reduce channel count for
  strictly computational reasons.

  Args:
    x: A 4-D tensorflow tensor.
    output_dim: An integer representing desired channel count in the output.
    training: If `True`, add the spectral norm assign ops.
    name: String to pass to the variable scope context.
  Returns:
    A new volume with the same batch, height, and width as the input.
  """
  with tf.compat.v1.variable_scope(
      name, custom_getter=sn_gettr(training=training)):
    w = tf.compat.v1.get_variable(
        'weights', [1, 1, x.get_shape()[-1], output_dim],
        initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform'))
    conv = tf.nn.conv2d(
        input=x, filters=w, strides=[1, 1, 1, 1], padding='SAME')
    return conv


def sn_non_local_block_sim(x, training=True, name='sn_nonlocal'):
  """Builds graph for the self-attention block.

  This is one third of the tricks from the SAGAN paper.

  Args:
    x: A 4-D tensorflow tensor.
    training: If `True`, add the spectral norm assign ops.
    name: String to pass to the variable scope context.
  Returns:
    A new volume with self-attention having been applied.
  """
  with tf.compat.v1.variable_scope(name):
    _, h, w, num_channels = x.shape.as_list()
    location_num = h * w
    downsampled_num = location_num // 4

    # theta path
    theta = sn_conv1x1(x, num_channels // 8, training, 'sn_conv_theta')
    theta = tf.reshape(
        theta, [-1, location_num, num_channels // 8])

    # phi path
    phi = sn_conv1x1(x, num_channels // 8, training, 'sn_conv_phi')
    phi = tf.compat.v1.layers.max_pooling2d(
        inputs=phi, pool_size=[2, 2], strides=2)
    phi = tf.reshape(
        phi, [-1, downsampled_num, num_channels // 8])

    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)

    # g path
    g = sn_conv1x1(x, num_channels // 2, training, 'sn_conv_g')
    g = tf.compat.v1.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
    g = tf.reshape(
        g, [-1, downsampled_num, num_channels // 2])

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [-1, h, w, num_channels // 2])
    sigma = tf.compat.v1.get_variable(
        'sigma_ratio', [], initializer=tf.compat.v1.initializers.constant(0.0))
    attn_g = sn_conv1x1(attn_g, num_channels, training, 'sn_conv_attn')
    return x + sigma * attn_g


@contextlib.contextmanager
def variables_on_gpu0():
  """Put variables on GPU."""
  old_fn = tf.compat.v1.get_variable

  def new_fn(*args, **kwargs):
    with tf.device('/gpu:0'):
      return old_fn(*args, **kwargs)

  tf.compat.v1.get_variable = new_fn
  yield
  tf.compat.v1.get_variable = old_fn


def avg_grads(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list is
      over individual gradients. The inner list is over the gradient calculation
      for each tower.

  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(input_tensor=grad, axis=0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
