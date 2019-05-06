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

"""Utilities to clip weights.

This is useful in the original formulation of the Wasserstein loss, which
requires that the discriminator be K-Lipschitz. See
https://arxiv.org/pdf/1701.07875 for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import tensorflow as tf

__all__ = [
    'clip_variables',
    'clip_discriminator_weights',
    'VariableClippingOptimizer',
]


def clip_discriminator_weights(optimizer, model, weight_clip):
  """Modifies an optimizer so it clips weights to a certain value.

  Args:
    optimizer: An optimizer to perform variable weight clipping.
    model: A GANModel namedtuple.
    weight_clip: Positive python float to clip discriminator weights. Used to
      enforce a K-lipschitz condition, which is useful for some GAN training
      schemes (ex WGAN: https://arxiv.org/pdf/1701.07875).

  Returns:
    An optimizer to perform weight clipping after updates.

  Raises:
    ValueError: If `weight_clip` is less than 0.
  """
  return clip_variables(optimizer, model.discriminator_variables, weight_clip)


def clip_variables(optimizer, variables, weight_clip):
  """Modifies an optimizer so it clips weights to a certain value.

  Args:
    optimizer: An optimizer to perform variable weight clipping.
    variables: A list of TensorFlow variables.
    weight_clip: Positive python float to clip discriminator weights. Used to
      enforce a K-lipschitz condition, which is useful for some GAN training
      schemes (ex WGAN: https://arxiv.org/pdf/1701.07875).

  Returns:
    An optimizer to perform weight clipping after updates.

  Raises:
    ValueError: If `weight_clip` is less than 0.
  """
  if weight_clip < 0:
    raise ValueError(
        '`discriminator_weight_clip` must be positive. Instead, was %s' %
        weight_clip)
  return VariableClippingOptimizer(
      opt=optimizer,
      # Do no reduction, so clipping happens per-value.
      vars_to_clip_dims={var: [] for var in variables},
      max_norm=weight_clip,
      use_locking=True,
      colocate_clip_ops_with_vars=True)


# Copied from
# `tensorflow/contrib/opt/python/training/variable_clipping_optimizer.py`.
class VariableClippingOptimizer(tf.compat.v1.train.Optimizer):
  """Wrapper optimizer that clips the norm of specified variables after update.

  This optimizer delegates all aspects of gradient calculation and application
  to an underlying optimizer.  After applying gradients, this optimizer then
  clips the variable to have a maximum L2 norm along specified dimensions.
  NB: this is quite different from clipping the norm of the gradients.

  Multiple instances of `VariableClippingOptimizer` may be chained to specify
  different max norms for different subsets of variables.

  This is more efficient at serving-time than using normalization during
  embedding lookup, at the expense of more expensive training and fewer
  guarantees about the norms.

  @@__init__

  """

  def __init__(self,
               opt,
               vars_to_clip_dims,
               max_norm,
               use_locking=False,
               colocate_clip_ops_with_vars=False,
               name='VariableClipping'):
    """Construct a new clip-norm optimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be one of the Optimizer classes.
      vars_to_clip_dims: A dict with keys as Variables and values as lists
        of dimensions along which to compute the L2-norm.  See
        `tf.clip_by_norm` for more details.
      max_norm: The L2-norm to clip to, for all variables specified.
      use_locking: If `True` use locks for clip update operations.
      colocate_clip_ops_with_vars: If `True`, try colocating the clip norm
        ops with the corresponding variable.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "VariableClipping".
    """
    super(VariableClippingOptimizer, self).__init__(use_locking, name)
    self._opt = opt
    # Defensive copy of input dict
    self._vars_to_clip_dims = {
        var: clip_dims[:] for var, clip_dims in vars_to_clip_dims.items()}
    self._max_norm = max_norm
    self._colocate_clip_ops_with_vars = colocate_clip_ops_with_vars

  def compute_gradients(self, *args, **kwargs):
    return self._opt.compute_gradients(*args, **kwargs)

  def get_slot(self, *args, **kwargs):
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    return self._opt.get_slot_names(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    with tf.compat.v1.name_scope(name, self._name) as name:
      update_op = self._opt.apply_gradients(
          grads_and_vars, global_step=global_step)
      clip_update_ops = []
      with tf.control_dependencies([update_op]):
        for grad, var in grads_and_vars:
          if grad is None or var not in self._vars_to_clip_dims:
            continue
          # `x.op` doesn't work in eager execution.
          suffix = var.name if tf.executing_eagerly() else var.op.name
          with tf.compat.v1.name_scope('clip_' + suffix):
            if isinstance(grad, tf.Tensor):
              clip_update_ops.append(self._clip_dense(var))
            else:
              clip_update_ops.append(self._clip_sparse(grad, var))

      # In case no var was clipped, still need to run the update_op.
      return tf.group(*([update_op] + clip_update_ops), name=name)

  def _clip_dense(self, var):
    with self._maybe_colocate_with(var):
      updated_var_value = var.read_value()
      normalized_var = tf.clip_by_norm(
          updated_var_value, self._max_norm, self._vars_to_clip_dims[var])
      delta = updated_var_value - normalized_var
    with tf.compat.v1.colocate_with(var):
      return var.assign_sub(delta, use_locking=self._use_locking)

  def _clip_sparse(self, grad, var):
    assert isinstance(grad, tf.IndexedSlices)
    clip_dims = self._vars_to_clip_dims[var]
    if 0 in clip_dims:
      # `x.op` doesn't work in eager execution.
      name = var.name if tf.executing_eagerly() else var.op.name
      tf.compat.v1.logging.warning(
          'Clipping norm across dims %s for %s is inefficient '
          'when including sparse dimension 0.', clip_dims, name)
      return self._clip_dense(var)

    with tf.compat.v1.colocate_with(var):
      var_subset = tf.gather(var, grad.indices)
    with self._maybe_colocate_with(var):
      normalized_var_subset = tf.clip_by_norm(
          var_subset, self._max_norm, clip_dims)
      delta = tf.IndexedSlices(
          var_subset - normalized_var_subset, grad.indices, grad.dense_shape)
    with tf.compat.v1.colocate_with(var):
      return var.scatter_sub(delta, use_locking=self._use_locking)

  @contextlib.contextmanager
  def _maybe_colocate_with(self, var):
    """Context to colocate with `var` if `colocate_clip_ops_with_vars`."""
    if self._colocate_clip_ops_with_vars:
      with tf.compat.v1.colocate_with(var):
        yield
    else:
      yield
