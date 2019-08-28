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

"""Tensorflow operations specific to TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

from tensorflow.python.tpu import tpu_function  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'cross_replica_mean',
    'cross_replica_moments',
]


def cross_replica_mean(inputs, group_size=None):
  """Calculates the average value of inputs tensor across TPU replicas."""
  num_replicas = tpu_function.get_tpu_context().number_of_shards
  if not group_size:
    group_size = num_replicas
  if group_size == 1:
    return inputs
  if group_size != num_replicas:
    group_assignment = []
    assert num_replicas % group_size == 0
    for g in range(num_replicas // group_size):
      replica_ids = [g * group_size + i for i in range(group_size)]
      group_assignment.append(replica_ids)
  else:
    group_assignment = None
  return tf.compat.v1.tpu.cross_replica_sum(inputs, group_assignment) / tf.cast(
      group_size, inputs.dtype)


def cross_replica_moments(inputs, axis, parallel=True, group_size=None):
  """Compute mean and variance of the inputs tensor across TPU replicas.

  Args:
    inputs: A tensor with 2 or more dimensions.
    axis: Array of ints. Axes along which to compute mean and variance.
    parallel: Use E[x^2] - (E[x])^2 to compute variance. This can be done
      in parallel to computing the mean and reducing the communication overhead.
    group_size: Integer, the number of replicas to compute moments arcoss.
      None or 0 will use all replicas (global).

  Returns:
    Two tensors with mean and variance.
  """
  # Compute local mean and then average across replicas.
  mean = tf.math.reduce_mean(input_tensor=inputs, axis=axis)
  mean = cross_replica_mean(mean)
  if parallel:
    # Compute variance using the E[x^2] - (E[x])^2 formula. This is less
    # numerically stable than the E[(x-E[x])^2] formula, but allows the two
    # cross-replica sums to be computed in parallel, saving communication
    # overhead.
    mean_of_squares = tf.reduce_mean(input_tensor=tf.square(inputs), axis=axis)
    mean_of_squares = cross_replica_mean(mean_of_squares, group_size=group_size)
    mean_squared = tf.square(mean)
    variance = mean_of_squares - mean_squared
  else:
    variance = tf.math.reduce_mean(
        input_tensor=tf.math.square(inputs - mean), axis=axis)
  variance = cross_replica_mean(variance, group_size=group_size)
  return mean, variance
