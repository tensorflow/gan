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

"""Tests for tfgan.features.virtual_batchnorm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.python.features.virtual_batchnorm import vbn_statistics


class VirtualBatchnormTest(tf.test.TestCase):

  def test_syntax(self):
    reference_batch = tf.zeros([5, 3, 16, 9, 15])
    vbn = tfgan.features.VBN(reference_batch, batch_axis=1)
    vbn(tf.ones([5, 7, 16, 9, 15]))

  def test_no_broadcast_needed(self):
    """When `axis` and `batch_axis` are at the end, no broadcast is needed."""
    reference_batch = tf.zeros([5, 3, 16, 9, 15])
    minibatch = tf.zeros([5, 3, 16, 3, 15])
    vbn = tfgan.features.VBN(reference_batch, axis=-1, batch_axis=-2)
    vbn(minibatch)

  def test_statistics(self):
    """Check that `_statistics` gives the same result as `nn.moments`."""
    tf.set_random_seed(1234)

    tensors = tf.random_normal([4, 5, 7, 3])
    for axes in [(3), (0, 2), (1, 2, 3)]:
      vb_mean, mean_sq = vbn_statistics(tensors, axes)
      mom_mean, mom_var = tf.nn.moments(tensors, axes)
      vb_var = mean_sq - tf.square(vb_mean)

      with self.cached_session(use_gpu=True) as sess:
        vb_mean_np, vb_var_np, mom_mean_np, mom_var_np = sess.run([
            vb_mean, vb_var, mom_mean, mom_var])

      self.assertAllClose(mom_mean_np, vb_mean_np)
      self.assertAllClose(mom_var_np, vb_var_np)

  def test_virtual_statistics(self):
    """Check that `_virtual_statistics` gives same result as `nn.moments`."""
    tf.set_random_seed(1234)

    batch_axis = 0
    partial_batch = tf.random_normal([4, 5, 7, 3])
    single_example = tf.random_normal([1, 5, 7, 3])
    full_batch = tf.concat([partial_batch, single_example], axis=0)

    for reduction_axis in range(1, 4):
      # Get `nn.moments` on the full batch.
      reduction_axes = list(range(4))
      del reduction_axes[reduction_axis]
      mom_mean, mom_variance = tf.nn.moments(full_batch, reduction_axes)

      # Get virtual batch statistics.
      vb_reduction_axes = list(range(4))
      del vb_reduction_axes[reduction_axis]
      del vb_reduction_axes[batch_axis]
      vbn = tfgan.features.VBN(partial_batch, reduction_axis)
      vb_mean, mean_sq = vbn._virtual_statistics(
          single_example, vb_reduction_axes)
      vb_variance = mean_sq - tf.square(vb_mean)
      # Remove singleton batch dim for easy comparisons.
      vb_mean = tf.squeeze(vb_mean, batch_axis)
      vb_variance = tf.squeeze(vb_variance, batch_axis)

      with self.cached_session(use_gpu=True) as sess:
        vb_mean_np, vb_var_np, mom_mean_np, mom_var_np = sess.run([
            vb_mean, vb_variance, mom_mean, mom_variance])

      self.assertAllClose(mom_mean_np, vb_mean_np)
      self.assertAllClose(mom_var_np, vb_var_np)

  def test_reference_batch_normalization(self):
    """Check that batch norm from VBN agrees with opensource implementation."""
    tf.set_random_seed(1234)

    batch = tf.random_normal([6, 5, 7, 3, 3])

    for axis in range(5):
      # Get `layers` batchnorm result.
      bn_normalized = tf.layers.batch_normalization(batch, axis, training=True)

      # Get VBN's batch normalization on reference batch.
      batch_axis = 0 if axis is not 0 else 1  # axis and batch_axis can't same
      vbn = tfgan.features.VBN(batch, axis, batch_axis=batch_axis)
      vbn_normalized = vbn.reference_batch_normalization()

      with self.cached_session(use_gpu=True) as sess:
        tf.global_variables_initializer().run()

        bn_normalized_np, vbn_normalized_np = sess.run(
            [bn_normalized, vbn_normalized])
      self.assertAllClose(bn_normalized_np, vbn_normalized_np)

  def test_same_as_batchnorm(self):
    """Check that batch norm on set X is the same as ref of X / y on `y`."""
    tf.set_random_seed(1234)

    num_examples = 4
    examples = [tf.random_normal([5, 7, 3]) for _ in range(num_examples)]

    # Get the result of the opensource batch normalization.
    batch_normalized = tf.layers.batch_normalization(
        tf.stack(examples), training=True)

    for i in range(num_examples):
      examples_except_i = tf.stack(examples[:i] + examples[i+1:])
      # Get the result of VBN's batch normalization.
      vbn = tfgan.features.VBN(examples_except_i)
      vb_normed = tf.squeeze(vbn(tf.expand_dims(examples[i], [0])), [0])

      with self.cached_session(use_gpu=True) as sess:
        tf.global_variables_initializer().run()
        bn_np, vb_np = sess.run([batch_normalized, vb_normed])
      self.assertAllClose(bn_np[i, ...], vb_np)

  def test_minibatch_independent(self):
    """Test that virtual batch normalized examples are independent.

    Unlike batch normalization, virtual batch normalization has the property
    that the virtual batch normalized value of an example is independent of the
    other examples in the minibatch. In this test, we verify this property.
    """
    tf.set_random_seed(1234)

    # These can be random, but must be the same for all session calls.
    reference_batch = tf.constant(
        np.random.normal(size=[4, 7, 3]), dtype=tf.float32)
    fixed_example = tf.constant(np.random.normal(size=[7, 3]), dtype=tf.float32)

    # Get the VBN object and the virtual batch normalized value for
    # `fixed_example`.
    vbn = tfgan.features.VBN(reference_batch)
    vbn_fixed_example = tf.squeeze(vbn(tf.expand_dims(fixed_example, 0)), 0)
    with self.session(use_gpu=True):
      tf.global_variables_initializer().run()
      vbn_fixed_example_np = vbn_fixed_example.eval()

    # Check that the value is the same for different minibatches, and different
    # sized minibatches.
    for minibatch_size in range(1, 6):
      examples = [tf.random_normal([7, 3]) for _ in range(minibatch_size)]

      minibatch = tf.stack([fixed_example] + examples)
      vbn_minibatch = vbn(minibatch)
      cur_vbn_fixed_example = vbn_minibatch[0, ...]
      with self.cached_session(use_gpu=True):
        tf.global_variables_initializer().run()
        cur_vbn_fixed_example_np = cur_vbn_fixed_example.eval()
      self.assertAllClose(vbn_fixed_example_np, cur_vbn_fixed_example_np)

  def test_variable_reuse(self):
    """Test that variable scopes work and inference on a real-ish case."""
    tensor1_ref = tf.zeros([6, 5, 7, 3, 3])
    tensor1_examples = tf.zeros([4, 5, 7, 3, 3])
    tensor2_ref = tf.zeros([4, 2, 3])
    tensor2_examples = tf.zeros([2, 2, 3])

    with tf.variable_scope('dummy_scope', reuse=True):
      with self.assertRaisesRegexp(
          ValueError, 'does not exist, or was not created with '
          'tf.get_variable()'):
        tfgan.features.VBN(tensor1_ref)

    vbn1 = tfgan.features.VBN(tensor1_ref, name='vbn1')
    vbn2 = tfgan.features.VBN(tensor2_ref, name='vbn2')

    # Fetch reference and examples after virtual batch normalization. Also
    # fetch in variable reuse case.
    to_fetch = []

    to_fetch.append(vbn1.reference_batch_normalization())
    to_fetch.append(vbn2.reference_batch_normalization())
    to_fetch.append(vbn1(tensor1_examples))
    to_fetch.append(vbn2(tensor2_examples))

    tf.get_variable_scope().reuse_variables()

    to_fetch.append(vbn1.reference_batch_normalization())
    to_fetch.append(vbn2.reference_batch_normalization())
    to_fetch.append(vbn1(tensor1_examples))
    to_fetch.append(vbn2(tensor2_examples))

    self.assertEqual(4, len(tf.contrib.framework.get_variables()))

    with self.session(use_gpu=True) as sess:
      tf.global_variables_initializer().run()
      sess.run(to_fetch)

  def test_invalid_input(self):
    # Reference batch has unknown dimensions.
    with self.assertRaisesRegexp(
        ValueError, '`reference_batch` has unknown dimensions.'):
      tfgan.features.VBN(tf.placeholder(tf.float32), name='vbn1')

    # Axis too negative.
    with self.assertRaisesRegexp(
        ValueError, 'Value of `axis` argument .* is out of range'):
      tfgan.features.VBN(tf.zeros([1, 2]), axis=-3, name='vbn2')

    # Axis too large.
    with self.assertRaisesRegexp(
        ValueError, 'Value of `axis` argument .* is out of range'):
      tfgan.features.VBN(tf.zeros([1, 2]), axis=2, name='vbn3')

    # Batch axis too negative.
    with self.assertRaisesRegexp(
        ValueError, 'Value of `axis` argument .* is out of range'):
      tfgan.features.VBN(tf.zeros([1, 2]), name='vbn4', batch_axis=-3)

    # Batch axis too large.
    with self.assertRaisesRegexp(
        ValueError, 'Value of `axis` argument .* is out of range'):
      tfgan.features.VBN(tf.zeros([1, 2]), name='vbn5', batch_axis=2)

    # Axis and batch axis are the same.
    with self.assertRaisesRegexp(
        ValueError, '`axis` and `batch_axis` cannot be the same.'):
      tfgan.features.VBN(tf.zeros(
          [1, 2]), axis=1, name='vbn6', batch_axis=1)

    # Reference Tensor and example Tensor have incompatible shapes.
    tensor_ref = tf.zeros([5, 2, 3])
    tensor_examples = tf.zeros([3, 2, 3])
    vbn = tfgan.features.VBN(tensor_ref, name='vbn7', batch_axis=1)
    with self.assertRaisesRegexp(ValueError, 'Shapes .* are incompatible'):
      vbn(tensor_examples)


if __name__ == '__main__':
  tf.test.main()
