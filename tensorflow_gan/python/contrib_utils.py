# coding=utf-8
# Copyright 2018 The TensorFlow GAN Authors.
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
  return get_variables(scope, suffix, tf.GraphKeys.TRAINABLE_VARIABLES)


def get_variables(scope=None, suffix=None,
                  collection=tf.GraphKeys.GLOBAL_VARIABLES):
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
  if isinstance(scope, tf.VariableScope):
    scope = scope.name
  if suffix is not None:
    if ':' not in suffix:
      suffix += ':'
    scope = (scope or '') + '.*' + suffix
  return tf.get_collection(collection, scope)
