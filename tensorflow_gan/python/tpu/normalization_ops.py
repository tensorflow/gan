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

"""Provides ops for supporting TPU operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from six.moves import range
import tensorflow as tf
from tensorflow_gan.python.tpu.cross_replica_ops import cross_replica_moments

from tensorflow.python.tpu import tpu_function  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.training import moving_averages  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'batch_norm',
    'standardize_batch',
]


def batch_norm(inputs,
               is_training,
               conditional_class_labels=None,
               axis=-1,
               variance_epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer=tf.compat.v1.initializers.zeros(),
               gamma_initializer=tf.compat.v1.initializers.ones(),
               batch_axis=0,
               name='batch_norm'):
  """Adds Batch Norm or Conditional Batch Norm.

  Args:
    inputs: Tensor of inputs (e.g. images).
    is_training: Whether or not the layer is in training mode. In training
      mode it would accumulate the statistics of the moments into the
      `moving_mean` and `moving_variance` using an exponential moving average
      with the given `decay`. When is_training=False, these variables are not
      updated, and the precomputed values are used verbatim.
    conditional_class_labels: If `None`, this layer is vanilla Batch
      Normalization. If not, it is a tensor of one-hot labels - same first
      dimension as inputs, and the layer is Conditional Batch Normalization
      with normalization constants determined by the class (see
      https://arxiv.org/pdf/1610.07629.pdf for more detail).
    axis: Integer, the axis that should be normalized (typically the features
        axis). For instance, after a `Convolution2D` layer with
        `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    variance_epsilon: A small float number to avoid dividing by 0.
    center: If True, add offset of `beta` to normalized tensor. If False,
      `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can
      be disabled since the scaling can be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    batch_axis: The axis of the batch dimension.
    name: name: String name to be used for scoping.
  Returns:
    Output tensor.
  """
  with tf.compat.v1.variable_scope(
      name, values=[inputs], reuse=tf.compat.v1.AUTO_REUSE):
    # Determine the variable shape.
    var_shape = [1] * inputs.shape.rank
    var_shape[axis] = tf.compat.dimension_value(inputs.shape[axis])
    # Allocate parameters for the trainable variables.
    if conditional_class_labels is not None:
      num_categories = tf.compat.dimension_value(
          conditional_class_labels.shape[-1])
      var_shape[batch_axis] = num_categories
      labels = tf.math.argmax(
          input=conditional_class_labels, axis=1)  # to integer
      if center:
        beta = tf.compat.v1.get_variable(
            'beta', var_shape, initializer=beta_initializer)
        beta = tf.gather(beta, labels)
      if scale:
        gamma = tf.compat.v1.get_variable(
            'gamma', var_shape, initializer=gamma_initializer)
        gamma = tf.gather(gamma, labels)
    else:
      if center:
        beta = tf.compat.v1.get_variable(
            'beta', var_shape, initializer=beta_initializer)
      if scale:
        gamma = tf.compat.v1.get_variable(
            'gamma', var_shape, initializer=gamma_initializer)
    outputs = standardize_batch(
        inputs, is_training=is_training, epsilon=variance_epsilon, offset=beta,
        scale=gamma)
    outputs.set_shape(inputs.shape)
    return outputs


def standardize_batch(inputs,
                      is_training,
                      offset=None,
                      scale=None,
                      decay=0.999,
                      epsilon=1e-3,
                      data_format='NHWC',
                      use_moving_averages=True,
                      use_cross_replica_mean=None):
  """Adds TPU-enabled batch normalization layer.

  Details on Batch Normalization can be found in 'Batch Normalization:
  Accelerating Deep Network Training by Reducing Internal Covariate Shift',
  Ioffe S. and Szegedy C. 2015 [http://arxiv.org/abs/1502.03167].

  Note #1: This method computes the batch statistic across all TPU replicas,
  thus simulating the true batch norm in the distributed setting. If one wants
  to avoid the cross-replica communication set use_cross_replica_mean=False.

  Note #2: When is_training is True the moving_mean and moving_variance need
  to be updated in each training step. By default, the update_ops are placed
  in `tf.GraphKeys.UPDATE_OPS` and they need to be added as a dependency to
  the `train_op`. For example:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops)
      total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  Note #3: Reasonable values for `decay` are close to 1.0, typically in the
  multiple-nines range: 0.999, 0.99, 0.9, etc. Lower the `decay` value (trying
  `decay`=0.9) if model experiences reasonably good training performance but
  poor validation and/or test performance.

  Args:
    inputs: A tensor with 2 or 4 dimensions, where the first dimension is
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC`, and the second dimension if `data_format` is
      `NCHW`.
    is_training: Whether or not the layer is in training mode. In training
      mode it would accumulate the statistics of the moments into the
      `moving_mean` and `moving_variance` using an exponential moving average
      with the given `decay`. When is_training=False, these variables are not
      updated, and the precomputed values are used verbatim.
    offset: An offset `Tensor`, often denoted `beta` in equations, or
      None. If present, will be added to the normalized tensor.
    scale: A scale `Tensor`, often denoted `gamma` in equations, or
      `None`. If present, the scale is applied to the normalized tensor.
    decay: Decay for the moving averages. See notes above for reasonable
      values.
    epsilon: Small float added to variance to avoid dividing by zero.
    data_format: Input data format. NHWC or NCHW.
    use_moving_averages: If True keep moving averages of mean and variance that
      are used during inference. Otherwise use accumlators.
    use_cross_replica_mean: If True add operations to do computes batch norm
      statistics across all TPU cores. These ops are not compatible with other
      platforms. The default (None) will only add the operations if running
      on TPU.

  Returns:
    The normalized tensor with the same type and shape as `inputs`.
  """
  if data_format not in {'NCHW', 'NHWC'}:
    raise ValueError(
        'Invalid data_format {}. Allowed: NCHW, NHWC.'.format(data_format))
  if use_cross_replica_mean is None:
    # Default to global batch norm only on TPUs.
    use_cross_replica_mean = (
        tpu_function.get_tpu_context().number_of_shards is not None)
    logging.debug('Automatically determined use_cross_replica_mean=%s.',
                  use_cross_replica_mean)

  inputs = tf.convert_to_tensor(value=inputs)
  inputs_dtype = inputs.dtype
  inputs_shape = inputs.get_shape()

  num_channels = tf.compat.dimension_value(inputs.shape[-1])
  if num_channels is None:
    raise ValueError('`C` dimension must be known but is None')

  inputs_rank = inputs_shape.ndims
  if inputs_rank is None:
    raise ValueError('Inputs %s has undefined rank' % inputs.name)
  elif inputs_rank not in [2, 4]:
    raise ValueError(
        'Inputs %s has unsupported rank.'
        ' Expected 2 or 4 but got %d' % (inputs.name, inputs_rank))
  # Bring 2-D inputs into 4-D format.
  if inputs_rank == 2:
    new_shape = [-1, 1, 1, num_channels]
    if data_format == 'NCHW':
      new_shape = [-1, num_channels, 1, 1]
    inputs = tf.reshape(inputs, new_shape)
    if offset is not None:
      offset = tf.reshape(offset, new_shape)
    if scale is not None:
      scale = tf.reshape(scale, new_shape)

  # Execute a distributed batch normalization
  axis = 1 if data_format == 'NCHW' else 3
  inputs = tf.cast(inputs, tf.float32)
  reduction_axes = [i for i in range(4) if i != axis]
  if use_cross_replica_mean:
    mean, variance = cross_replica_moments(inputs, reduction_axes)
  else:
    counts, mean_ss, variance_ss, _ = tf.nn.sufficient_statistics(
        inputs, reduction_axes, keepdims=False)
    mean, variance = tf.nn.normalize_moments(
        counts, mean_ss, variance_ss, shift=None)

  if use_moving_averages:
    mean, variance = moving_moments_for_inference(
        mean=mean, variance=variance, is_training=is_training, decay=decay)
  else:
    mean, variance = accumulated_moments_for_inference(
        mean=mean, variance=variance, is_training=is_training)

  outputs = tf.nn.batch_normalization(
      inputs,
      mean=mean,
      variance=variance,
      offset=offset,
      scale=scale,
      variance_epsilon=epsilon)
  outputs = tf.cast(outputs, inputs_dtype)
  # Bring 2-D inputs back into 2-D format.
  if inputs_rank == 2:
    outputs = tf.reshape(outputs, [-1] + inputs_shape[1:].as_list())
  outputs.set_shape(inputs_shape)
  return outputs


def moving_moments_for_inference(mean, variance, is_training, decay):
  """Use moving averages of moments during inference.

  Args:
    mean: Tensor of shape [num_channels] with the mean of the current batch.
    variance: Tensor of shape [num_channels] with the variance of the current
      batch.
    is_training: Boolean, wheather to construct ops for training or inference
      graph.
    decay: Decay rate to use for moving averages.

  Returns:
    Tuple of (mean, variance) to use. This can the same as the inputs.
  """
  # Create the moving average variables and add them to the appropriate
  # collections.
  variable_collections = [
      tf.compat.v1.GraphKeys.MOVING_AVERAGE_VARIABLES,
      tf.compat.v1.GraphKeys.MODEL_VARIABLES,
      tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
  ]
  # Disable partition setting for moving_mean and moving_variance
  # as assign_moving_average op below doesn"t support partitioned variable.
  moving_mean = tf.compat.v1.get_variable(
      'moving_mean',
      shape=mean.shape,
      initializer=tf.compat.v1.zeros_initializer(),
      trainable=False,
      partitioner=None,
      collections=variable_collections)
  moving_variance = tf.compat.v1.get_variable(
      'moving_variance',
      shape=variance.shape,
      initializer=tf.compat.v1.ones_initializer(),
      trainable=False,
      partitioner=None,
      collections=variable_collections)
  if is_training:
    logging.debug('Adding update ops for moving averages of mean and variance.')
    # Update variables for mean and variance during training.
    update_moving_mean = moving_averages.assign_moving_average(
        moving_mean,
        tf.cast(mean, moving_mean.dtype),
        decay,
        zero_debias=False)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance,
        tf.cast(variance, moving_variance.dtype),
        decay,
        zero_debias=False)
    tf.compat.v1.add_to_collection(
        tf.compat.v1.GraphKeys.UPDATE_OPS,
        tf.group(
            update_moving_mean, update_moving_variance, name='ema_update_ops'))
    return mean, variance
  logging.debug('Using moving mean and variance.')
  return moving_mean, moving_variance


def accumulated_moments_for_inference(mean, variance, is_training):
  """Use accumulated statistics for moments during inference.

  After training the user is responsible for filling the accumulators with the
  actual values.

  Args:
    mean: Tensor of shape [num_channels] with the mean of the current batch.
    variance: Tensor of shape [num_channels] with the variance of the current
      batch.
    is_training: Boolean, wheather to construct ops for training or inference
      graph.

  Returns:
    Tuple of (mean, variance) to use. This can the same as the inputs.
  """
  variable_collections = [
      tf.compat.v1.GraphKeys.MODEL_VARIABLES,
      tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
  ]
  with tf.compat.v1.variable_scope('accu', values=[mean, variance]):
    # Create variables for accumulating batch statistic and use them during
    # inference. The ops for filling the accumulators must be created and run
    # before eval. See docstring above.
    accu_mean = tf.compat.v1.get_variable(
        'accu_mean',
        shape=mean.shape,
        initializer=tf.compat.v1.zeros_initializer(),
        trainable=False,
        collections=variable_collections)
    accu_variance = tf.compat.v1.get_variable(
        'accu_variance',
        shape=variance.shape,
        initializer=tf.compat.v1.zeros_initializer(),
        trainable=False,
        collections=variable_collections)
    accu_counter = tf.compat.v1.get_variable(
        'accu_counter',
        shape=[],
        initializer=tf.compat.v1.initializers.constant(1e-12),
        trainable=False,
        collections=variable_collections)
    update_accus = tf.compat.v1.get_variable(
        'update_accus',
        shape=[],
        dtype=tf.int32,
        initializer=tf.compat.v1.zeros_initializer(),
        trainable=False,
        collections=variable_collections)

    mean = tf.identity(mean, 'mean')
    variance = tf.identity(variance, 'variance')

    if is_training:
      return mean, variance

    logging.debug('Using accumulated moments.')
    # Return the accumulated batch statistics and add current batch statistics
    # to accumulators if update_accus variables equals 1.
    def update_accus_fn():
      return tf.group([
          tf.compat.v1.assign_add(accu_mean, mean),
          tf.compat.v1.assign_add(accu_variance, variance),
          tf.compat.v1.assign_add(accu_counter, 1),
      ])

    dep = tf.cond(
        pred=tf.equal(update_accus, 1),
        true_fn=update_accus_fn,
        false_fn=tf.no_op)
    with tf.control_dependencies([dep]):
      return accu_mean / accu_counter, accu_variance / accu_counter
