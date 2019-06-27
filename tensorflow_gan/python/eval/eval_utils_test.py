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

"""Tests for tfgan.eval.eval_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_gan.python.eval import eval_utils
import tensorflow_probability as tfp


class UtilsTest(tf.test.TestCase):

  def test_image_grid(self):
    eval_utils.image_grid(
        input_tensor=tf.zeros([25, 32, 32, 3]),
        grid_shape=(5, 5))

  def test_python_image_grid(self):
    image_grid = eval_utils.python_image_grid(
        input_array=np.zeros([25, 32, 32, 3]),
        grid_shape=(5, 5))
    self.assertTupleEqual(image_grid.shape, (5 * 32, 5 * 32, 3))

  # TODO(joelshor): Add more `image_reshaper` tests.
  def test_image_reshaper_image_list(self):
    images = eval_utils.image_reshaper(
        images=tf.unstack(tf.zeros([25, 32, 32, 3])),
        num_cols=2)
    images.shape.assert_is_compatible_with([1, 13 * 32, 2 * 32, 3])

  def test_image_reshaper_image(self):
    images = eval_utils.image_reshaper(
        images=tf.zeros([25, 32, 32, 3]),
        num_cols=2)
    images.shape.assert_is_compatible_with([1, 13 * 32, 2 * 32, 3])


class StreamingUtilsTest(tf.test.TestCase):

  def test_mean_correctness(self):
    """Checks value of streaming_mean_tensor_float64."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 8
    data = np.random.randn(num_batches, 3, 4, 5)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(3, 4, 5))
    value, update_op = eval_utils.streaming_mean_tensor_float64(placeholder)

    expected_result = np.mean(data, axis=0)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        sess.run(update_op, feed_dict={placeholder: data[i]})
      result = sess.run(value)
      self.assertAllClose(expected_result, result, rtol=1e-15, atol=1e-15)

  def test_mean_update_op_value(self):
    """Checks that the value of the update op is the same as the value."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 8
    data = np.random.randn(num_batches, 3, 4, 5)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(3, 4, 5))
    value, update_op = eval_utils.streaming_mean_tensor_float64(placeholder)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        update_op_value = sess.run(update_op, feed_dict={placeholder: data[i]})
        result = sess.run(value)
        self.assertAllClose(update_op_value, result)

  def test_mean_float32(self):
    """Checks handling of float32 tensors in streaming_mean_tensor_float64."""
    if tf.executing_eagerly():
      # streaming_mean_tensor_float64 is not supported when eager execution is
      # enabled.
      return
    data = tf.constant([1., 2., 3.], tf.float32)
    value, update_op = eval_utils.streaming_mean_tensor_float64(data)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose([1., 2., 3.], update_op)
      self.assertAllClose([1., 2., 3.], value)

  def test_covariance_simple(self):
    """Sanity check for streaming_covariance."""
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.]]
    result, update_op = eval_utils.streaming_covariance(
        tf.constant(x, dtype=tf.float64))
    expected_result = np.cov(x, rowvar=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)

  def test_covariance_with_y(self):
    """Checks output of streaming_covariance given value for y."""
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.]]
    y = [[3., 3.], [1., 0.]]
    result, update_op = eval_utils.streaming_covariance(
        x=tf.constant(x, dtype=tf.float64),
        y=tf.constant(y, dtype=tf.float64))
    # We mulitiply by N/(N-1)=2 to get the unbiased estimator.
    # Note that we cannot use np.cov since it uses a different semantics for y.
    expected_result = 2. * tfp.stats.covariance(x, y)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)

  def test_covariance_float32(self):
    """Checks handling of float32 values in streaming_covariance."""
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.]]
    result, update_op = eval_utils.streaming_covariance(
        x=tf.constant(x, dtype=tf.float32))
    expected_result = np.cov(x, rowvar=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)

  def test_covariance_float32_with_y(self):
    """Checks handling of float32 values in streaming_covariance."""
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.]]
    y = [[1., 2.], [2., 1.]]
    result, update_op = eval_utils.streaming_covariance(
        x=tf.constant(x, dtype=tf.float32),
        y=tf.constant(y, dtype=tf.float32))
    # We mulitiply by N/(N-1)=2 to get the unbiased estimator.
    # Note that we cannot use np.cov since it uses a different semantics for y.
    expected_result = 2. * tfp.stats.covariance(x, y)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)

  def test_covariance_batches(self):
    """Checks value consistency of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 8
    data = np.random.randn(num_batches, 4, 5)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(4, 5))
    value, update_op = eval_utils.streaming_covariance(placeholder)

    expected_result = np.cov(
        np.reshape(data, [num_batches * 4, 5]), rowvar=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        update_op_result = sess.run(update_op, feed_dict={placeholder: data[i]})
        result = sess.run(value)
        self.assertAllClose(update_op_result, result)
      self.assertAllClose(expected_result, result)

  def test_covariance_accuracy(self):
    """Checks accuracy of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 128
    batch_size = 32
    dim = 32
    data = np.random.randn(num_batches, batch_size, dim)

    placeholder = tf.compat.v1.placeholder(
        dtype=tf.float64, shape=(batch_size, dim))
    value, update_op = eval_utils.streaming_covariance(placeholder)

    expected_result = np.cov(
        np.reshape(data, [num_batches * batch_size, dim]), rowvar=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        sess.run(update_op, feed_dict={placeholder: data[i]})
      result = sess.run(value)
      self.assertAllClose(expected_result, result, rtol=1e-15, atol=1e-15)

  def test_covariance_accuracy_with_y(self):
    """Checks accuracy of streaming_covariance with two input tensors."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 128
    batch_size = 32
    dim = 32
    x = np.random.randn(num_batches, batch_size, dim)
    y = np.random.randn(num_batches, batch_size, dim)

    placeholder_x = tf.compat.v1.placeholder(
        dtype=tf.float64, shape=(batch_size, dim))
    placeholder_y = tf.compat.v1.placeholder(
        dtype=tf.float64, shape=(batch_size, dim))
    value, update_op = eval_utils.streaming_covariance(placeholder_x,
                                                       placeholder_y)

    # We mulitiply by N/(N-1) to get the unbiased estimator.
    # Note that we cannot use np.cov since it uses different semantics for y.
    expected_result = num_batches * batch_size / (
        num_batches * batch_size - 1) * tfp.stats.covariance(
            x=np.reshape(x, [num_batches * batch_size, dim]),
            y=np.reshape(y, [num_batches * batch_size, dim]))
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        sess.run(
            update_op, feed_dict={
                placeholder_x: x[i],
                placeholder_y: y[i]
            })
      result = sess.run(value)
      self.assertAllClose(expected_result, result, rtol=1e-15, atol=1e-15)


if __name__ == '__main__':
  tf.test.main()
