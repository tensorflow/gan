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

"""Utilities for removing or replacing contrib functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_trainable_variables(scope=None, suffix=None):
  """Gets the list of trainable variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a list of variables in the trainable collection with scope and suffix.
  """
  return get_variables(scope, suffix,
                       tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)


def get_variables_by_name(given_name, scope=None):
  """Gets the list of variables that were given that name.

  Args:
    given_name: name given to the variable without any scope.
    scope: an optional scope for filtering the variables to return.

  Returns:
    a copied list of variables with the given name and scope.
  """
  suffix = '/' + given_name + ':|^' + given_name + ':'
  return get_variables(scope=scope, suffix=suffix)


def _with_dependencies(dependencies, output_tensor):
  with tf.compat.v1.name_scope(
      'control_dependency', values=list(dependencies) + [output_tensor]):
    with tf.compat.v1.colocate_with(output_tensor), tf.control_dependencies(
        dependencies):
      output_tensor = tf.convert_to_tensor(value=output_tensor)
      assert isinstance(output_tensor, tf.Tensor)
      return tf.identity(output_tensor)


def get_variables(scope=None,
                  suffix=None,
                  collection=tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
  """Gets the list of variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return. Can be a
      variable scope or a string.
    suffix: an optional suffix for filtering the variables to return.
    collection: in which collection search for. Defaults to
      `GraphKeys.GLOBAL_VARIABLES`.

  Returns:
    a list of variables in collection with scope and suffix.
  """
  if isinstance(scope, tf.compat.v1.VariableScope):
    scope = scope.name
  if suffix is not None:
    if ':' not in suffix:
      suffix += ':'
    scope = (scope or '') + '.*' + suffix
  return tf.compat.v1.get_collection(collection, scope)


_USE_GLOBAL_STEP = 0


def create_train_op(total_loss,
                    optimizer,
                    global_step=_USE_GLOBAL_STEP,
                    update_ops=None,
                    variables_to_train=None,
                    transform_grads_fn=None,
                    summarize_gradients=False,
                    gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False,
                    check_numerics=True):
  """Creates an `Operation` that evaluates the gradients and returns the loss.

  Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `_USE_GLOBAL_STEP`, then tf.train.global_step() is used.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
      a warning will be displayed.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.trainable_variables().
    transform_grads_fn: A function which takes a single argument, a list of
      gradient to variable pairs (tuples), performs any requested gradient
      updates, such as gradient clipping or multipliers, and returns the updated
      list.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.
    check_numerics: Whether or not we apply check_numerics.

  Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
  """
  if global_step is _USE_GLOBAL_STEP:
    global_step = tf.compat.v1.train.get_or_create_global_step()

  # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
  global_update_ops = set(
      tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
  if update_ops is None:
    update_ops = global_update_ops
  else:
    update_ops = set(update_ops)
  if not global_update_ops.issubset(update_ops):
    tf.compat.v1.logging.warning(
        'update_ops in create_train_op does not contain all the '
        'update_ops in GraphKeys.UPDATE_OPS')

  # Make sure update_ops are computed before total_loss.
  if update_ops:
    with tf.control_dependencies(update_ops):
      barrier = tf.no_op(name='update_barrier')
    total_loss = _with_dependencies([barrier], total_loss)

  if variables_to_train is None:
    # Default to tf.trainable_variables()
    variables_to_train = tf.compat.v1.trainable_variables()
  else:
    # Make sure that variables_to_train are in tf.trainable_variables()
    for v in variables_to_train:
      assert v in tf.compat.v1.trainable_variables()

  assert variables_to_train

  # Create the gradients. Note that apply_gradients adds the gradient
  # computation to the current graph.
  grads = optimizer.compute_gradients(
      total_loss,
      variables_to_train,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  if transform_grads_fn:
    grads = transform_grads_fn(grads)

  # Summarize gradients.
  if summarize_gradients:
    with tf.compat.v1.name_scope('summarize_grads'):
      add_gradients_summaries(grads)

  # Create gradient updates.
  grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  with tf.compat.v1.name_scope('train_op'):
    # Make sure total_loss is valid.
    if check_numerics:
      total_loss = tf.debugging.check_numerics(total_loss,
                                               'LossTensor is inf or nan')

    # Ensure the train_tensor computes grad_updates.

    train_op = _with_dependencies([grad_updates], total_loss)

  # Add the operation used for training to the 'train_op' collection
  train_ops = tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.TRAIN_OP)
  if train_op not in train_ops:
    train_ops.append(train_op)

  return train_op


def add_gradients_summaries(grads_and_vars):
  """Add summaries to gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).

  Returns:
    The list of created summaries.
  """
  summaries = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad
      summaries.append(
          tf.compat.v1.summary.histogram(var.op.name + '_gradient',
                                         grad_values))
      summaries.append(
          tf.compat.v1.summary.scalar(var.op.name + '_gradient_norm',
                                      tf.linalg.global_norm([grad_values])))
    else:
      tf.compat.v1.logging.info('Var %s has no gradient', var.op.name)

  return summaries


def batch_to_space(*args, **kwargs):
  try:
    return tf.batch_to_space(*args, **kwargs)
  except TypeError:
    if 'block_shape' in kwargs:
      kwargs['block_size'] = kwargs['block_shape']
      del kwargs['block_shape']
    return tf.batch_to_space(*args, **kwargs)
