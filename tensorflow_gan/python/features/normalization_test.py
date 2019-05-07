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

"""Tests for features.normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

import tensorflow as tf

from tensorflow_gan.python import contrib_utils
from tensorflow_gan.python.features import normalization as norm


class InstanceNormTest(tf.test.TestCase, absltest.TestCase):

  def testUnknownShape(self):
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    inputs = tf.compat.v1.placeholder(tf.float32)
    with self.assertRaisesRegexp(ValueError, 'undefined rank'):
      norm.instance_norm(inputs)

  def testBadDataFormat(self):
    inputs = tf.zeros((2, 5, 5), dtype=tf.float32)
    with self.assertRaisesRegexp(ValueError,
                                 'data_format has to be either NCHW or NHWC.'):
      norm.instance_norm(inputs, data_format='NHCW')

  def testParamsShapeNotFullyDefinedNCHW(self):
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    inputs = tf.compat.v1.placeholder(tf.float32, shape=(3, None, 4))
    with self.assertRaisesRegexp(ValueError, 'undefined channels dimension'):
      norm.instance_norm(inputs, data_format='NCHW')

  def testParamsShapeNotFullyDefinedNHWC(self):
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    inputs = tf.compat.v1.placeholder(tf.float32, shape=(3, 4, None))
    with self.assertRaisesRegexp(ValueError, 'undefined channels dimension'):
      norm.instance_norm(inputs, data_format='NHWC')

  def testCreateOp(self):
    height, width = 3, 3
    images = tf.random.uniform((5, height, width, 3), seed=1)
    output = norm.instance_norm(images)
    self.assertListEqual([5, height, width, 3], output.shape.as_list())

  def testCreateOpFloat64(self):
    height, width = 3, 3
    images = tf.random.uniform((5, height, width, 3), dtype=tf.float64, seed=1)
    output = norm.instance_norm(images)
    self.assertListEqual([5, height, width, 3], output.shape.as_list())

  def testCreateOpNoScaleCenter(self):
    if tf.executing_eagerly():
      # Collections don't work with eager.
      return
    height, width = 3, 3
    images = tf.random.uniform((5, height, width, 3), dtype=tf.float64, seed=1)
    output = norm.instance_norm(images, center=False, scale=False)
    self.assertListEqual([5, height, width, 3], output.shape.as_list())
    self.assertEmpty(contrib_utils.get_variables_by_name('beta'))
    self.assertEmpty(contrib_utils.get_variables_by_name('gamma'))

  def testCreateVariables(self):
    if tf.executing_eagerly():
      # Collections don't work with eager.
      return
    height, width = 3, 3
    images = tf.random.uniform((5, height, width, 3), seed=1)
    norm.instance_norm(images, center=True, scale=True)
    self.assertLen(contrib_utils.get_variables_by_name('beta'), 1)
    self.assertLen(contrib_utils.get_variables_by_name('gamma'), 1)

  def testReuseVariables(self):
    if tf.executing_eagerly():
      # Variable reuse doesn't work with eager.
      return
    height, width = 3, 3
    images = tf.random.uniform((5, height, width, 3), seed=1)
    norm.instance_norm(images, scale=True, scope='IN')
    norm.instance_norm(images, scale=True, scope='IN', reuse=True)
    self.assertLen(contrib_utils.get_variables_by_name('beta'), 1)
    self.assertLen(contrib_utils.get_variables_by_name('gamma'), 1)

  def testValueCorrectWithReuseVars(self):
    height, width = 3, 3
    image_shape = (10, height, width, 3)
    images = tf.random.uniform(image_shape, seed=1)
    output_train = norm.instance_norm(images, scope='IN')
    output_eval = norm.instance_norm(images, scope='IN', reuse=True)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      # output_train and output_eval should be the same.
      train_np, eval_np = sess.run([output_train, output_eval])
      self.assertAllClose(train_np, eval_np)

  def doOutputTest(self, input_shape, data_format, tol=1e-3):
    axis = -1 if data_format == 'NHWC' else 1
    for mu in (0.0, 1e2):
      for sigma in (1.0, 0.1):
        # Determine shape of Tensor after norm.
        reduced_shape = (input_shape[0], input_shape[axis])
        expected_mean = np.zeros(reduced_shape)
        expected_var = np.ones(reduced_shape)

        # Determine axes that will be normalized.
        reduced_axes = list(range(len(input_shape)))
        del reduced_axes[axis]
        del reduced_axes[0]
        reduced_axes = tuple(reduced_axes)

        inputs = tf.random.uniform(input_shape, seed=0) * sigma + mu
        output_op = norm.instance_norm(
            inputs, center=False, scale=False, data_format=data_format)
        with self.cached_session() as sess:
          sess.run(tf.compat.v1.global_variables_initializer())
          outputs = sess.run(output_op)
          # Make sure that there are no NaNs
          self.assertFalse(np.isnan(outputs).any())
          mean = np.mean(outputs, axis=reduced_axes)
          var = np.var(outputs, axis=reduced_axes)
          # The mean and variance of each example should be close to 0 and 1
          # respectively.
          self.assertAllClose(expected_mean, mean, rtol=tol, atol=tol)
          self.assertAllClose(expected_var, var, rtol=tol, atol=tol)

  def testOutputSmallInput4DNHWC(self):
    self.doOutputTest((10, 10, 10, 30), 'NHWC', tol=1e-2)

  def testOutputSmallInput4DNCHW(self):
    self.doOutputTest((10, 10, 10, 30), 'NCHW', tol=1e-2)

  def testOutputBigInput4DNHWC(self):
    self.doOutputTest((1, 100, 100, 1), 'NHWC', tol=1e-3)

  def testOutputBigInput4DNCHW(self):
    self.doOutputTest((1, 100, 100, 1), 'NCHW', tol=1e-3)

  def testOutputSmallInput5DNHWC(self):
    self.doOutputTest((10, 10, 10, 10, 30), 'NHWC', tol=1e-2)

  def testOutputSmallInput5DNCHW(self):
    self.doOutputTest((10, 10, 10, 10, 30), 'NCHW', tol=1e-2)

  def testOutputBigInput5DNHWC(self):
    self.doOutputTest((1, 100, 100, 1, 1), 'NHWC', tol=1e-3)

  def testOutputBigInput5DNCHW(self):
    self.doOutputTest((1, 100, 100, 1, 1), 'NCHW', tol=1e-3)


class GroupNormTest(tf.test.TestCase, absltest.TestCase):

  def testInvalidGroupSize(self):
    inputs = tf.zeros((5, 2, 10, 10), dtype=tf.float32)
    with self.assertRaisesRegexp(ValueError,
                                 'Invalid groups 10 for 2 channels.'):
      norm.group_norm(
          inputs, groups=10, reduction_axes=[-2, -1], channels_axis=-3)

  def testBadCommensurateGroup(self):
    inputs = tf.zeros((5, 4, 10, 10), dtype=tf.float32)
    with self.assertRaisesRegexp(ValueError,
                                 '4 channels is not commensurate with '
                                 '3 groups.'):
      norm.group_norm(
          inputs, groups=3, reduction_axes=[-2, -1], channels_axis=-3)

  def testAxisIsBad(self):
    inputs = tf.zeros((1, 2, 4, 5), dtype=tf.float32)
    with self.assertRaisesRegexp(ValueError,
                                 'Axis is out of bounds.'):
      norm.group_norm(inputs, channels_axis=5)
    with self.assertRaisesRegexp(ValueError,
                                 'Axis is out of bounds.'):
      norm.group_norm(inputs, reduction_axes=[1, 5])

  def testNotMutuallyExclusiveAxis(self):
    inputs = tf.zeros((10, 32, 32, 32), dtype=tf.float32)
    # Specify axis with negative values.
    with self.assertRaisesRegexp(ValueError, 'mutually exclusive'):
      norm.group_norm(inputs, channels_axis=-2, reduction_axes=[-2])
    # Specify axis with positive values.
    with self.assertRaisesRegexp(ValueError, 'mutually exclusive'):
      norm.group_norm(inputs, channels_axis=1, reduction_axes=[1, 3])
    # Specify axis with mixed positive and negative values.
    with self.assertRaisesRegexp(ValueError, 'mutually exclusive'):
      norm.group_norm(inputs, channels_axis=-2, reduction_axes=[2])

  def testUnknownShape(self):
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    inputs = tf.compat.v1.placeholder(tf.float32)
    with self.assertRaisesRegexp(ValueError, 'undefined rank'):
      norm.group_norm(inputs)

  def testParamsShapeNotFullyDefinedReductionAxes(self):
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    inputs = tf.compat.v1.placeholder(tf.float32, shape=(1, 32, None, 4))
    with self.assertRaisesRegexp(ValueError, 'undefined dimensions'):
      norm.group_norm(inputs)

  def testParamsShapeNotFullyDefinedChannelsAxis(self):
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    inputs = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 4, None))
    with self.assertRaisesRegexp(ValueError, 'undefined channel dimension'):
      norm.group_norm(inputs, channels_axis=-1, reduction_axes=[-3, -2])

  def testParamsShapeNotFullyDefinedBatchAxis(self):
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    height, width, groups = 3, 3, 4
    inputs = tf.compat.v1.placeholder(
        tf.float32, shape=(None, height, width, 2 * groups))
    output = norm.group_norm(
        inputs, channels_axis=-1, reduction_axes=[-3, -2], groups=groups)
    self.assertListEqual([None, height, width, 2 * groups],
                         output.shape.as_list())

  def testCreateOp(self):
    height, width, groups = 3, 3, 4
    images = tf.random.uniform((5, height, width, 2 * groups), seed=1)
    output = norm.group_norm(
        images, groups=groups, channels_axis=-1, reduction_axes=[-3, -2])
    self.assertListEqual([5, height, width, 2*groups], output.shape.as_list())

  def testCreateOpFloat64(self):
    height, width, groups = 3, 3, 5
    images = tf.random.uniform((5, height, width, 4 * groups),
                               dtype=tf.float64,
                               seed=1)
    output = norm.group_norm(images, groups=groups)
    self.assertEqual(tf.float64, output.dtype)
    self.assertListEqual([5, height, width, 4*groups], output.shape.as_list())

  def testCreateOpNoScaleCenter(self):
    if tf.executing_eagerly():
      # Collections don't work with eager.
      return
    height, width, groups = 3, 3, 7
    images = tf.random.uniform((5, height, width, 3 * groups),
                               dtype=tf.float32,
                               seed=1)
    output = norm.group_norm(images, groups=groups, center=False, scale=False)
    self.assertListEqual([5, height, width, 3*groups], output.shape.as_list())
    self.assertEmpty(contrib_utils.get_variables_by_name('beta'))
    self.assertEmpty(contrib_utils.get_variables_by_name('gamma'))

  def testCreateVariables_NHWC(self):
    if tf.executing_eagerly():
      # Collections don't work with eager.
      return
    height, width = 3, 3
    images = tf.random.uniform((5, height, width, 8), seed=1)
    norm.group_norm(
        images,
        groups=4,
        channels_axis=-1,
        reduction_axes=(-3, -2),
        center=True,
        scale=True)
    self.assertLen(contrib_utils.get_variables_by_name('beta'), 1)
    self.assertLen(contrib_utils.get_variables_by_name('gamma'), 1)

  def testCreateVariables_NCHW(self):
    if tf.executing_eagerly():
      # Collections don't work with eager.
      return
    height, width, groups = 3, 3, 4
    images = tf.random.uniform((5, 2 * groups, height, width), seed=1)
    norm.group_norm(
        images,
        groups=4,
        channels_axis=-3,
        reduction_axes=(-2, -1),
        center=True,
        scale=True)
    self.assertLen(contrib_utils.get_variables_by_name('beta'), 1)
    self.assertLen(contrib_utils.get_variables_by_name('gamma'), 1)

  def testReuseVariables(self):
    if tf.executing_eagerly():
      # Variable reuse doesn't work with eager.
      return
    height, width = 3, 3
    images = tf.random.uniform((5, height, width, 4), seed=1)
    norm.group_norm(images, groups=2, scale=True, scope='IN')
    norm.group_norm(images, groups=2, scale=True, scope='IN', reuse=True)
    self.assertLen(contrib_utils.get_variables_by_name('beta'), 1)
    self.assertLen(contrib_utils.get_variables_by_name('gamma'), 1)

  def testValueCorrectWithReuseVars(self):
    height, width = 3, 3
    image_shape = (10, height, width, 4)
    images = tf.random.uniform(image_shape, seed=1)
    output_train = norm.group_norm(images, groups=2, scope='IN')
    output_eval = norm.group_norm(images, groups=2, scope='IN', reuse=True)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      # output_train and output_eval should be the same.
      train_np, eval_np = sess.run([output_train, output_eval])
      self.assertAllClose(train_np, eval_np)

  def doOutputTest(self,
                   input_shape,
                   channels_axis=None,
                   reduction_axes=None,
                   mean_close_to_zero=False,
                   groups=2,
                   tol=1e-2):
    # Select the axis for the channel and the dimensions along which statistics
    # are accumulated.
    if channels_axis < 0:
      channels_axis += len(input_shape)
    reduced_axes = [channels_axis + 1]
    for a in reduction_axes:
      if a < 0:
        a += len(input_shape)
      if a < channels_axis:
        reduced_axes.append(a)
      else:
        reduced_axes.append(a+1)
    reduced_axes = tuple(reduced_axes)

    # Calculate the final shape for the output Tensor.
    axes_before_channels = input_shape[:channels_axis]
    axes_after_channels = input_shape[channels_axis+1:]
    channels = input_shape[channels_axis]
    outputs_shape = (axes_before_channels + [groups, channels // groups] +
                     axes_after_channels)

    # Calculate the final shape for the output statistics.
    reduced_shape = []
    for i, a in enumerate(outputs_shape):
      if i not in reduced_axes:
        reduced_shape.append(a)

    if mean_close_to_zero:
      mu_tuple = (1e-4, 1e-2, 1.0)
      sigma_tuple = (1e-2, 0.1, 1.0)
    else:
      mu_tuple = (1.0, 1e2)
      sigma_tuple = (1.0, 0.1)

    for mu in mu_tuple:
      for sigma in sigma_tuple:
        # Determine shape of Tensor after norm.
        expected_mean = np.zeros(reduced_shape)
        expected_var = np.ones(reduced_shape)

        inputs = tf.random.normal(input_shape, seed=0) * sigma + mu
        output_op = norm.group_norm(
            inputs,
            groups=groups,
            center=False,
            scale=False,
            channels_axis=channels_axis,
            reduction_axes=reduction_axes,
            mean_close_to_zero=mean_close_to_zero)
        with self.cached_session() as sess:
          sess.run(tf.compat.v1.global_variables_initializer())
          outputs = sess.run(output_op)
          # Make sure that there are no NaNs
          self.assertFalse(np.isnan(outputs).any())

          outputs = np.reshape(outputs, outputs_shape)
          mean = np.mean(outputs, axis=reduced_axes)
          var = np.var(outputs, axis=reduced_axes)
          # The mean and variance of each example should be close to 0 and 1
          # respectively.
          self.assertAllClose(expected_mean, mean, rtol=tol, atol=tol)
          self.assertAllClose(expected_var, var, rtol=tol, atol=tol)

  def doOutputTestForMeanCloseToZero(self,
                                     input_shape,
                                     channels_axis=None,
                                     reduction_axes=None,
                                     groups=2,
                                     tol=5e-2):
    self.doOutputTest(
        input_shape,
        channels_axis=channels_axis,
        reduction_axes=reduction_axes,
        groups=groups,
        tol=tol,
        mean_close_to_zero=True)

  def testOutputSmallInput4D_NHWC(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])
    # Specify axes with positive values.
    self.doOutputTestForMeanCloseToZero(
        input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doOutputTestForMeanCloseToZero(
        input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutputSmallInput3D_NHWC(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])
    # Specify axes with positive values.
    self.doOutputTestForMeanCloseToZero(
        input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doOutputTestForMeanCloseToZero(
        input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutputSmallInput4D_NCHW(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])
    # Specify axes with positive values.
    self.doOutputTestForMeanCloseToZero(
        input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doOutputTestForMeanCloseToZero(
        input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  def testOutputSmallInput3D_NCHW(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=0, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])
    # Specify axes with positive values.
    self.doOutputTestForMeanCloseToZero(
        input_shape, channels_axis=0, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doOutputTestForMeanCloseToZero(
        input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  def testOutputBigInput4D_NHWC(self):
    self.doOutputTest(
        [5, 100, 100, 1], channels_axis=3, reduction_axes=[1, 2], groups=1)
    self.doOutputTestForMeanCloseToZero(
        [5, 100, 100, 1], channels_axis=3, reduction_axes=[1, 2], groups=1)

  def testOutputBigInput4D_NCHW(self):
    self.doOutputTest(
        [1, 100, 100, 4], channels_axis=1, reduction_axes=[2, 3], groups=4)
    self.doOutputTestForMeanCloseToZero(
        [1, 100, 100, 4], channels_axis=1, reduction_axes=[2, 3], groups=4)

  def testOutputSmallInput2D_NC(self):
    self.doOutputTest(
        [10, 7 * 100], channels_axis=1, reduction_axes=[], groups=7)
    self.doOutputTestForMeanCloseToZero(
        [10, 7 * 100], channels_axis=1, reduction_axes=[], groups=7)

  def testOutputSmallInput5D_NCXXX(self):
    self.doOutputTest(
        [10, 10, 20, 40, 5],
        channels_axis=1,
        reduction_axes=[2, 3, 4],
        groups=5)
    self.doOutputTestForMeanCloseToZero(
        [10, 10, 20, 40, 5],
        channels_axis=1,
        reduction_axes=[2, 3, 4],
        groups=5)


if __name__ == '__main__':
  tf.test.main()
