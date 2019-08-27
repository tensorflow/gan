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

"""Utilities for running tests in a TF 1.x and 2.x compatible way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def crop_and_resize(*args, **kwargs):
  """`tf.image.crop_and_resize` that works for TF 1.x and 2.x."""
  try:
    return tf.image.crop_and_resize(*args, **kwargs)
  except (TypeError, AttributeError):
    if 'box_ind' in kwargs:
      kwargs['box_indices'] = kwargs['box_ind']
      del kwargs['box_ind']
    return tf.image.crop_and_resize(*args, **kwargs)


def nn_avg_pool2d(*args, **kwargs):
  """`tf.nn.avg_pool2d` that works for TF 1.x and 2.x."""
  try:
    return tf.nn.avg_pool2d(*args, **kwargs)
  except (TypeError, AttributeError):
    if 'input' in kwargs:
      kwargs['value'] = kwargs['input']
      del kwargs['input']
    return tf.nn.avg_pool(*args, **kwargs)


def batch_to_space(*args, **kwargs):
  """`tf.batch_to_space` that works for TF 1.x and 2.x."""
  try:
    return tf.batch_to_space(*args, **kwargs)
  except TypeError:
    if 'block_shape' in kwargs:
      kwargs['block_size'] = kwargs['block_shape']
      del kwargs['block_shape']
    return tf.batch_to_space(*args, **kwargs)
