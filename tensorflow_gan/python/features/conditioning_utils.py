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

"""Miscellaneous utilities for TFGAN code and examples.

Includes:
1) Conditioning the value of a Tensor, based on techniques from
  https://arxiv.org/abs/1609.03499.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


__all__ = [
    'condition_tensor',
    'condition_tensor_from_onehot',
]


def _get_shape(tensor):
  tensor_shape = tf.shape(input=tensor)
  static_tensor_shape = tf.get_static_value(tensor_shape)
  if static_tensor_shape is None:
    return tensor_shape
  else:
    return static_tensor_shape


def condition_tensor(tensor, conditioning):
  """Condition the value of a tensor.

  Conditioning scheme based on https://arxiv.org/abs/1609.03499.

  Args:
    tensor: A minibatch tensor to be conditioned.
    conditioning: A minibatch Tensor of to condition on. Must be 2D, with first
      dimension the same as `tensor`.

  Returns:
    `tensor` conditioned on `conditioning`.

  Raises:
    ValueError: If the non-batch dimensions of `tensor` aren't fully defined.
    ValueError: If `conditioning` isn't at least 2D.
    ValueError: If the batch dimension for the input Tensors don't match.
  """
  tensor.shape[1:].assert_is_fully_defined()
  num_features = tensor.shape[1:].num_elements()
  if conditioning.shape.ndims < 2:
    raise ValueError('conditioning must be at least 2D, but saw shape: %s'
                     % conditioning.shape)

  mapped_conditioning = tf.compat.v1.layers.dense(
      tf.compat.v1.layers.flatten(conditioning),
      num_features,
      kernel_initializer=tf.compat.v1.glorot_uniform_initializer())
  if not mapped_conditioning.shape.is_compatible_with(tensor.shape):
    mapped_conditioning = tf.reshape(mapped_conditioning, _get_shape(tensor))
  return tensor + mapped_conditioning


def _one_hot_to_embedding(one_hot, embedding_size):
  """Get a dense embedding vector from a one-hot encoding."""
  num_tokens = one_hot.shape[1]
  label_id = tf.argmax(input=one_hot, axis=1)
  embedding = tf.compat.v1.get_variable(
      'embedding', [num_tokens, embedding_size])
  return tf.nn.embedding_lookup(
      params=embedding, ids=label_id, name='token_to_embedding')


def _validate_onehot(one_hot_labels):
  one_hot_labels.shape.assert_has_rank(2)
  one_hot_labels.shape[1:].assert_is_fully_defined()


def condition_tensor_from_onehot(tensor, one_hot_labels, embedding_size=256):
  """Condition a tensor based on a one-hot tensor.

  Conditioning scheme based on https://arxiv.org/abs/1609.03499.

  Args:
    tensor: Tensor to be conditioned.
    one_hot_labels: A Tensor of one-hot labels. Shape is
      [batch_size, num_classes].
    embedding_size: The size of the class embedding.

  Returns:
    `tensor` conditioned on `one_hot_labels`.

  Raises:
    ValueError: `one_hot_labels` isn't 2D, if non-batch dimensions aren't
      fully defined, or if batch sizes don't match.
  """
  _validate_onehot(one_hot_labels)

  conditioning = _one_hot_to_embedding(one_hot_labels, embedding_size)
  return condition_tensor(tensor, conditioning)
