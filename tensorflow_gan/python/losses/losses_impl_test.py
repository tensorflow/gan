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

"""Tests for tfgan.losses.wargs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan.python.losses.losses_impl import numerically_stable_global_norm
import tensorflow_probability as tfp


class _LossesTest(object):

  def init_constants(self):
    self._discriminator_real_outputs_np = [-5.0, 1.4, 12.5, 2.7]
    self._discriminator_gen_outputs_np = [10.0, 4.4, -5.5, 3.6]
    self._weights = 2.3
    self._discriminator_real_outputs = tf.constant(
        self._discriminator_real_outputs_np, dtype=tf.float32)
    self._discriminator_gen_outputs = tf.constant(
        self._discriminator_gen_outputs_np, dtype=tf.float32)

  def test_generator_all_correct(self):
    loss = self._g_loss_fn(self._discriminator_gen_outputs)
    self.assertEqual(self._discriminator_gen_outputs.dtype, loss.dtype)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_g_loss, sess.run(loss), 5)

  def test_discriminator_all_correct(self):
    loss = self._d_loss_fn(
        self._discriminator_real_outputs, self._discriminator_gen_outputs)
    self.assertEqual(self._discriminator_gen_outputs.dtype, loss.dtype)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_d_loss, sess.run(loss), 5)

  def test_generator_loss_collection(self):
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    self.assertEmpty(tf.compat.v1.get_collection('collection'))
    self._g_loss_fn(
        self._discriminator_gen_outputs, loss_collection='collection')
    self.assertLen(tf.compat.v1.get_collection('collection'), 1)

  def test_discriminator_loss_collection(self):
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    self.assertEmpty(tf.compat.v1.get_collection('collection'))
    self._d_loss_fn(
        self._discriminator_real_outputs, self._discriminator_gen_outputs,
        loss_collection='collection')
    self.assertLen(tf.compat.v1.get_collection('collection'), 1)

  def test_generator_no_reduction(self):
    loss = self._g_loss_fn(
        self._discriminator_gen_outputs,
        reduction=tf.compat.v1.losses.Reduction.NONE)
    self.assertAllEqual([4], loss.shape)

  def test_discriminator_no_reduction(self):
    loss = self._d_loss_fn(
        self._discriminator_real_outputs,
        self._discriminator_gen_outputs,
        reduction=tf.compat.v1.losses.Reduction.NONE)
    self.assertAllEqual([4], loss.shape)

  def test_generator_patch(self):
    loss = self._g_loss_fn(
        tf.reshape(self._discriminator_gen_outputs, [2, 2]))
    self.assertEqual(self._discriminator_gen_outputs.dtype, loss.dtype)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_g_loss, sess.run(loss), 5)

  def test_discriminator_patch(self):
    loss = self._d_loss_fn(
        tf.reshape(self._discriminator_real_outputs, [2, 2]),
        tf.reshape(self._discriminator_gen_outputs, [2, 2]))
    self.assertEqual(self._discriminator_gen_outputs.dtype, loss.dtype)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_d_loss, sess.run(loss), 5)

  def test_generator_loss_with_placeholder_for_logits(self):
    if tf.executing_eagerly():
      # Eager execution doesn't support placeholders.
      return
    logits = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
    weights = tf.ones_like(logits, dtype=tf.float32)

    loss = self._g_loss_fn(logits, weights=weights)
    self.assertEqual(logits.dtype, loss.dtype)

    with self.cached_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          logits: [[10.0, 4.4, -5.5, 3.6]],
                      })
      self.assertAlmostEqual(self._expected_g_loss, loss, 5)

  def test_discriminator_loss_with_placeholder_for_logits(self):
    if tf.executing_eagerly():
      # Eager execution doesn't support placeholders.
      return
    logits = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
    logits2 = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
    real_weights = tf.ones_like(logits, dtype=tf.float32)
    generated_weights = tf.ones_like(logits, dtype=tf.float32)

    loss = self._d_loss_fn(
        logits, logits2, real_weights=real_weights,
        generated_weights=generated_weights)

    with self.cached_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          logits: [self._discriminator_real_outputs_np],
                          logits2: [self._discriminator_gen_outputs_np],
                      })
      self.assertAlmostEqual(self._expected_d_loss, loss, 5)

  def test_generator_with_python_scalar_weight(self):
    loss = self._g_loss_fn(
        self._discriminator_gen_outputs, weights=self._weights)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_g_loss * self._weights,
                             sess.run(loss), 4)

  def test_discriminator_with_python_scalar_weight(self):
    loss = self._d_loss_fn(
        self._discriminator_real_outputs, self._discriminator_gen_outputs,
        real_weights=self._weights, generated_weights=self._weights)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_d_loss * self._weights,
                             sess.run(loss), 4)

  def test_generator_with_scalar_tensor_weight(self):
    loss = self._g_loss_fn(self._discriminator_gen_outputs,
                           weights=tf.constant(self._weights))
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_g_loss * self._weights,
                             sess.run(loss), 4)

  def test_discriminator_with_scalar_tensor_weight(self):
    weights = tf.constant(self._weights)
    loss = self._d_loss_fn(
        self._discriminator_real_outputs, self._discriminator_gen_outputs,
        real_weights=weights, generated_weights=weights)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_d_loss * self._weights,
                             sess.run(loss), 4)

  def test_generator_add_summaries(self):
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    self.assertEmpty(
        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))
    self._g_loss_fn(self._discriminator_gen_outputs, add_summaries=True)
    self.assertLess(
        0, len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))

  def test_discriminator_add_summaries(self):
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    self.assertEmpty(
        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))
    self._d_loss_fn(
        self._discriminator_real_outputs, self._discriminator_gen_outputs,
        add_summaries=True)
    self.assertLess(
        0, len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))


class LeastSquaresLossTest(tf.test.TestCase, absltest.TestCase, _LossesTest):
  """Tests for least_squares_xxx_loss."""

  def setUp(self):
    super(LeastSquaresLossTest, self).setUp()
    self.init_constants()
    self._expected_g_loss = 17.69625
    self._expected_d_loss = 41.73375
    self._g_loss_fn = tfgan.losses.wargs.least_squares_generator_loss
    self._d_loss_fn = tfgan.losses.wargs.least_squares_discriminator_loss


class ModifiedLossTest(tf.test.TestCase, absltest.TestCase, _LossesTest):
  """Tests for modified_xxx_loss."""

  def setUp(self):
    super(ModifiedLossTest, self).setUp()
    self.init_constants()
    self._expected_g_loss = 1.38582
    self._expected_d_loss = 6.19637
    self._g_loss_fn = tfgan.losses.wargs.modified_generator_loss
    self._d_loss_fn = tfgan.losses.wargs.modified_discriminator_loss


class MinimaxLossTest(tf.test.TestCase, absltest.TestCase, _LossesTest):
  """Tests for minimax_xxx_loss."""

  def setUp(self):
    super(MinimaxLossTest, self).setUp()
    self.init_constants()
    self._expected_g_loss = -4.82408
    self._expected_d_loss = 6.19637
    self._g_loss_fn = tfgan.losses.wargs.minimax_generator_loss
    self._d_loss_fn = tfgan.losses.wargs.minimax_discriminator_loss


class WassersteinLossTest(tf.test.TestCase, absltest.TestCase, _LossesTest):
  """Tests for wasserstein_xxx_loss."""

  def setUp(self):
    super(WassersteinLossTest, self).setUp()
    self.init_constants()
    self._expected_g_loss = -3.12500
    self._expected_d_loss = 0.22500
    self._g_loss_fn = tfgan.losses.wargs.wasserstein_generator_loss
    self._d_loss_fn = tfgan.losses.wargs.wasserstein_discriminator_loss


class HingeWassersteinLossTest(tf.test.TestCase, absltest.TestCase,
                               _LossesTest):
  """Tests for wasserstein_hinge_xxx_loss."""

  def setUp(self):
    super(HingeWassersteinLossTest, self).setUp()
    self.init_constants()
    self._expected_g_loss = -3.12500
    self._expected_d_loss = 6.7500
    self._g_loss_fn = tfgan.losses.wargs.wasserstein_hinge_generator_loss
    self._d_loss_fn = tfgan.losses.wargs.wasserstein_hinge_discriminator_loss


# TODO(joelshor): Refactor this test to use the same code as the other losses.
class ACGANLossTest(tf.test.TestCase, absltest.TestCase):
  """Tests for acgan_loss."""

  def setUp(self):
    super(ACGANLossTest, self).setUp()
    self._g_loss_fn = tfgan.losses.wargs.acgan_generator_loss
    self._d_loss_fn = tfgan.losses.wargs.acgan_discriminator_loss
    self._discriminator_gen_classification_logits_np = [[10.0, 4.4, -5.5, 3.6],
                                                        [-4.0, 4.4, 5.2, 4.6],
                                                        [1.1, 2.4, -3.5, 5.6],
                                                        [1.1, 2.4, -3.5, 5.6]]
    self._discriminator_real_classification_logits_np = [[-2.0, 0.4, 12.5, 2.7],
                                                         [-1.2, 1.9, 12.3, 2.6],
                                                         [-2.4, -1.7, 2.5, 2.7],
                                                         [1.1, 2.4, -3.5, 5.6]]
    self._one_hot_labels_np = [[0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [1, 0, 0, 0],
                               [1, 0, 0, 0]]
    self._weights = 2.3

    self._discriminator_gen_classification_logits = tf.constant(
        self._discriminator_gen_classification_logits_np, dtype=tf.float32)
    self._discriminator_real_classification_logits = tf.constant(
        self._discriminator_real_classification_logits_np, dtype=tf.float32)
    self._one_hot_labels = tf.constant(
        self._one_hot_labels_np, dtype=tf.float32)
    self._generator_kwargs = {
        'discriminator_gen_classification_logits':
        self._discriminator_gen_classification_logits,
        'one_hot_labels': self._one_hot_labels,
    }
    self._discriminator_kwargs = {
        'discriminator_gen_classification_logits':
        self._discriminator_gen_classification_logits,
        'discriminator_real_classification_logits':
        self._discriminator_real_classification_logits,
        'one_hot_labels': self._one_hot_labels,
    }
    self._expected_g_loss = 3.84974
    self._expected_d_loss = 9.43950

  def test_generator_all_correct(self):
    loss = self._g_loss_fn(**self._generator_kwargs)
    self.assertEqual(
        self._discriminator_gen_classification_logits.dtype, loss.dtype)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_g_loss, sess.run(loss), 5)

  def test_discriminator_all_correct(self):
    loss = self._d_loss_fn(**self._discriminator_kwargs)
    self.assertEqual(
        self._discriminator_gen_classification_logits.dtype, loss.dtype)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_d_loss, sess.run(loss), 5)

  def test_generator_loss_collection(self):
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    self.assertEmpty(tf.compat.v1.get_collection('collection'))
    self._g_loss_fn(loss_collection='collection', **self._generator_kwargs)
    self.assertLen(tf.compat.v1.get_collection('collection'), 1)

  def test_discriminator_loss_collection(self):
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    self.assertEmpty(tf.compat.v1.get_collection('collection'))
    self._d_loss_fn(loss_collection='collection', **self._discriminator_kwargs)
    self.assertLen(tf.compat.v1.get_collection('collection'), 1)

  def test_generator_no_reduction(self):
    loss = self._g_loss_fn(
        reduction=tf.compat.v1.losses.Reduction.NONE, **self._generator_kwargs)
    self.assertAllEqual([4], loss.shape)

  def test_discriminator_no_reduction(self):
    loss = self._d_loss_fn(
        reduction=tf.compat.v1.losses.Reduction.NONE,
        **self._discriminator_kwargs)
    self.assertAllEqual([4], loss.shape)

  def test_generator_patch(self):
    patch_args = {x: tf.reshape(y, [2, 2, 4]) for x, y in
                  self._generator_kwargs.items()}
    loss = self._g_loss_fn(**patch_args)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_g_loss, sess.run(loss), 5)

  def test_discriminator_patch(self):
    patch_args = {x: tf.reshape(y, [2, 2, 4]) for x, y in
                  self._discriminator_kwargs.items()}
    loss = self._d_loss_fn(**patch_args)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_d_loss, sess.run(loss), 5)

  def test_generator_loss_with_placeholder_for_logits(self):
    if tf.executing_eagerly():
      # Eager execution doesn't support placeholders.
      return
    gen_logits = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
    one_hot_labels = tf.compat.v1.placeholder(tf.int32, shape=(None, 4))

    loss = self._g_loss_fn(gen_logits, one_hot_labels)
    with self.cached_session() as sess:
      loss = sess.run(
          loss, feed_dict={
              gen_logits: self._discriminator_gen_classification_logits_np,
              one_hot_labels: self._one_hot_labels_np,
          })
      self.assertAlmostEqual(self._expected_g_loss, loss, 5)

  def test_discriminator_loss_with_placeholder_for_logits_and_weights(self):
    if tf.executing_eagerly():
      # Eager execution doesn't support placeholders.
      return
    gen_logits = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
    real_logits = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
    one_hot_labels = tf.compat.v1.placeholder(tf.int32, shape=(None, 4))

    loss = self._d_loss_fn(gen_logits, real_logits, one_hot_labels)

    with self.cached_session() as sess:
      loss = sess.run(
          loss, feed_dict={
              gen_logits: self._discriminator_gen_classification_logits_np,
              real_logits: self._discriminator_real_classification_logits_np,
              one_hot_labels: self._one_hot_labels_np,
          })
      self.assertAlmostEqual(self._expected_d_loss, loss, 5)

  def test_generator_with_python_scalar_weight(self):
    loss = self._g_loss_fn(weights=self._weights, **self._generator_kwargs)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_g_loss * self._weights,
                             sess.run(loss), 4)

  def test_discriminator_with_python_scalar_weight(self):
    loss = self._d_loss_fn(
        real_weights=self._weights, generated_weights=self._weights,
        **self._discriminator_kwargs)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_d_loss * self._weights,
                             sess.run(loss), 4)

  def test_generator_with_scalar_tensor_weight(self):
    loss = self._g_loss_fn(
        weights=tf.constant(self._weights), **self._generator_kwargs)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_g_loss * self._weights,
                             sess.run(loss), 4)

  def test_discriminator_with_scalar_tensor_weight(self):
    weights = tf.constant(self._weights)
    loss = self._d_loss_fn(real_weights=weights, generated_weights=weights,
                           **self._discriminator_kwargs)
    with self.cached_session() as sess:
      self.assertAlmostEqual(self._expected_d_loss * self._weights,
                             sess.run(loss), 4)

  def test_generator_add_summaries(self):
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    self.assertEmpty(
        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))
    self._g_loss_fn(add_summaries=True, **self._generator_kwargs)
    self.assertLess(
        0, len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))

  def test_discriminator_add_summaries(self):
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    self.assertEmpty(
        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))
    self._d_loss_fn(add_summaries=True, **self._discriminator_kwargs)
    self.assertLess(
        0, len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))


class _PenaltyTest(object):

  def test_all_correct(self):
    loss = self._penalty_fn(**self._kwargs)
    self.assertEqual(self._expected_dtype, loss.dtype)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertAlmostEqual(self._expected_loss, sess.run(loss), 6)

  def test_loss_collection(self):
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    self.assertLen(tf.compat.v1.get_collection('collection'), 0)
    self._penalty_fn(loss_collection='collection', **self._kwargs)
    self.assertLen(tf.compat.v1.get_collection('collection'), 1)

  def test_no_reduction(self):
    loss = self._penalty_fn(
        reduction=tf.compat.v1.losses.Reduction.NONE, **self._kwargs)
    self.assertAllEqual([self._batch_size], loss.shape)

  def test_python_scalar_weight(self):
    loss = self._penalty_fn(weights=2.3, **self._kwargs)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertAlmostEqual(self._expected_loss * 2.3, sess.run(loss), 3)

  def test_scalar_tensor_weight(self):
    loss = self._penalty_fn(weights=tf.constant(2.3), **self._kwargs)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertAlmostEqual(self._expected_loss * 2.3, sess.run(loss), 3)


class GradientPenaltyTest(tf.test.TestCase, absltest.TestCase, _PenaltyTest):
  """Tests for wasserstein_gradient_penalty."""

  def setUp(self):
    super(GradientPenaltyTest, self).setUp()
    self._penalty_fn = tfgan.losses.wargs.wasserstein_gradient_penalty
    self._generated_data_np = [[3.1, 2.3, -12.3, 32.1]]
    self._real_data_np = [[-12.3, 23.2, 16.3, -43.2]]
    self._expected_dtype = tf.float32

    with tf.compat.v1.variable_scope('fake_scope') as self._scope:
      self._discriminator_fn(0.0, 0.0)

    self._kwargs = {
        'generated_data': tf.constant(
            self._generated_data_np, dtype=self._expected_dtype),
        'real_data': tf.constant(
            self._real_data_np, dtype=self._expected_dtype),
        'generator_inputs': None,
        'discriminator_fn': self._discriminator_fn,
        'discriminator_scope': self._scope,
    }
    self._expected_loss = 9.00000
    self._expected_op_name = 'wasserstein_gradient_penalty/value'
    self._batch_size = 1

  def _discriminator_fn(self, inputs, _):
    tf.compat.v1.add_to_collection('fake_update_ops', tf.constant(1.0))
    return tf.compat.v1.get_variable('dummy_d', initializer=2.0) * inputs

  def test_all_correct(self):
    if tf.executing_eagerly():
      with self.assertRaises(RuntimeError):
        super(GradientPenaltyTest, self).test_all_correct()
    else:
      super(GradientPenaltyTest, self).test_all_correct()

  def test_no_reduction(self):
    if tf.executing_eagerly():
      with self.assertRaises(RuntimeError):
        super(GradientPenaltyTest, self).test_no_reduction()
    else:
      super(GradientPenaltyTest, self).test_no_reduction()

  def test_python_scalar_weight(self):
    if tf.executing_eagerly():
      with self.assertRaises(RuntimeError):
        super(GradientPenaltyTest, self).test_python_scalar_weight()
    else:
      super(GradientPenaltyTest, self).test_python_scalar_weight()

  def test_scalar_tensor_weight(self):
    if tf.executing_eagerly():
      with self.assertRaises(RuntimeError):
        super(GradientPenaltyTest, self).test_scalar_tensor_weight()
    else:
      super(GradientPenaltyTest, self).test_scalar_tensor_weight()

  def test_loss_with_placeholder(self):
    if tf.executing_eagerly():
      # Eager execution doesn't support placeholders.
      return
    generated_data = tf.compat.v1.placeholder(tf.float32, shape=(None, None))
    real_data = tf.compat.v1.placeholder(tf.float32, shape=(None, None))

    loss = tfgan.losses.wargs.wasserstein_gradient_penalty(
        generated_data,
        real_data,
        self._kwargs['generator_inputs'],
        self._kwargs['discriminator_fn'],
        self._kwargs['discriminator_scope'])
    self.assertEqual(generated_data.dtype, loss.dtype)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      loss = sess.run(loss,
                      feed_dict={
                          generated_data: self._generated_data_np,
                          real_data: self._real_data_np,
                      })
      self.assertAlmostEqual(self._expected_loss, loss, 5)

  def test_loss_using_one_sided_mode(self):
    if tf.executing_eagerly():
      # `tf.gradients` doesn't work in eager execution.
      return
    loss = tfgan.losses.wargs.wasserstein_gradient_penalty(
        tf.constant(self._generated_data_np, dtype=tf.float32),
        tf.constant(self._real_data_np, dtype=tf.float32),
        self._kwargs['generator_inputs'],
        self._kwargs['discriminator_fn'],
        self._kwargs['discriminator_scope'],
        one_sided=True)
    self.assertEqual(tf.float32, loss.dtype)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      loss = sess.run(loss)
      self.assertAlmostEqual(self._expected_loss, loss, 5)

  def test_loss_with_gradient_norm_target(self):
    """Test loss value with non default gradient norm target."""
    if tf.executing_eagerly():
      # `tf.gradients` doesn't work in eager execution.
      return
    loss = tfgan.losses.wargs.wasserstein_gradient_penalty(
        self._generated_data_np,
        self._real_data_np,
        self._kwargs['generator_inputs'],
        self._kwargs['discriminator_fn'],
        self._kwargs['discriminator_scope'],
        target=2.0)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      loss = sess.run(loss)
      self.assertAlmostEqual(1.0, loss, 5)

  def test_reuses_scope(self):
    """Test that gradient penalty reuses discriminator scope."""
    if tf.executing_eagerly():
      # `tf.gradients` doesn't work in eager execution.
      return
    num_vars = len(
        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))
    tfgan.losses.wargs.wasserstein_gradient_penalty(**self._kwargs)
    self.assertLen(
        tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES),
        num_vars)

  def test_works_with_get_collection(self):
    """Tests that gradient penalty works inside other scopes."""
    if tf.executing_eagerly():
      # Collections don't work in eager.
      return
    # We ran the discriminator once in the setup, so there should be an op
    # already in the collection.
    self.assertLen(
        tf.compat.v1.get_collection('fake_update_ops',
                                    self._kwargs['discriminator_scope'].name),
        1)

    # Make sure the op is added to the collection even if it's in a name scope.
    with tf.compat.v1.name_scope('loss'):
      tfgan.losses.wargs.wasserstein_gradient_penalty(**self._kwargs)
    self.assertLen(
        tf.compat.v1.get_collection('fake_update_ops',
                                    self._kwargs['discriminator_scope'].name),
        2)

    # Make sure the op is added to the collection even if it's in a variable
    # scope.
    with tf.compat.v1.variable_scope('loss_vscope'):
      tfgan.losses.wargs.wasserstein_gradient_penalty(**self._kwargs)
    self.assertLen(
        tf.compat.v1.get_collection('fake_update_ops',
                                    self._kwargs['discriminator_scope'].name),
        3)


class MutualInformationPenaltyTest(tf.test.TestCase, absltest.TestCase,
                                   _PenaltyTest):
  """Tests for mutual_information_penalty."""

  def setUp(self):
    super(MutualInformationPenaltyTest, self).setUp()
    self._penalty_fn = tfgan.losses.wargs.mutual_information_penalty
    self._structured_generator_inputs = [1.0, 2.0]
    self._predicted_distributions = [
        tfp.distributions.Categorical(logits=[1.0, 2.0]),
        tfp.distributions.Normal([0.0], [1.0]),
    ]
    self._expected_dtype = tf.float32

    self._kwargs = {
        'structured_generator_inputs': self._structured_generator_inputs,
        'predicted_distributions': self._predicted_distributions,
    }
    self._expected_loss = 1.61610
    self._expected_op_name = 'mutual_information_loss/mul_1'
    self._batch_size = 2


class CombineAdversarialLossTest(tf.test.TestCase):
  """Tests for combine_adversarial_loss."""

  def setUp(self):
    super(CombineAdversarialLossTest, self).setUp()
    self._generated_data_np = [[3.1, 2.3, -12.3, 32.1]]
    self._real_data_np = [[-12.3, 23.2, 16.3, -43.2]]
    self._generated_data = tf.constant(
        self._generated_data_np, dtype=tf.float32)
    self._real_data = tf.constant(
        self._real_data_np, dtype=tf.float32)
    self._generated_inputs = None
    self._expected_loss = 9.00000

  def _test_correct_helper(self, use_weight_factor):
    variable_list = [tf.Variable(1.0)]
    main_loss = variable_list[0] * 2
    adversarial_loss = variable_list[0] * 3
    gradient_ratio_epsilon = 1e-6
    if use_weight_factor:
      weight_factor = tf.constant(2.0)
      gradient_ratio = None
      adv_coeff = 2.0
      expected_loss = 1.0 * 2 + adv_coeff * 1.0 * 3
    else:
      weight_factor = None
      gradient_ratio = tf.constant(0.5)
      adv_coeff = 2.0 / (3 * 0.5 + gradient_ratio_epsilon)
      expected_loss = 1.0 * 2 + adv_coeff * 1.0 * 3
    combined_loss = tfgan.losses.wargs.combine_adversarial_loss(
        main_loss,
        adversarial_loss,
        weight_factor=weight_factor,
        gradient_ratio=gradient_ratio,
        gradient_ratio_epsilon=gradient_ratio_epsilon,
        variables=variable_list)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertNear(expected_loss, sess.run(combined_loss), 1e-5)

  def test_correct_useweightfactor(self):
    if tf.executing_eagerly():
      with self.assertRaises(RuntimeError):
        self._test_correct_helper(True)
    else:
      self._test_correct_helper(True)

  def test_correct_nouseweightfactor(self):
    if tf.executing_eagerly():
      with self.assertRaises(RuntimeError):
        self._test_correct_helper(False)
    else:
      self._test_correct_helper(False)

  def _test_no_weight_skips_adversarial_loss_helper(self, use_weight_factor):
    """Test the 0 adversarial weight or grad ratio skips adversarial loss."""
    main_loss = tf.constant(1.0)
    adversarial_loss = tf.constant(1.0)

    weight_factor = 0.0 if use_weight_factor else None
    gradient_ratio = None if use_weight_factor else 0.0

    combined_loss = tfgan.losses.wargs.combine_adversarial_loss(
        main_loss,
        adversarial_loss,
        weight_factor=weight_factor,
        gradient_ratio=gradient_ratio,
        gradient_summaries=False)

    with self.cached_session() as sess:
      self.assertEqual(1.0, sess.run(combined_loss))

  def test_no_weight_skips_adversarial_loss_useweightfactor(self):
    self._test_no_weight_skips_adversarial_loss_helper(True)

  def test_no_weight_skips_adversarial_loss_nouseweightfactor(self):
    if tf.executing_eagerly():
      with self.assertRaises(RuntimeError):
        self._test_no_weight_skips_adversarial_loss_helper(False)
    else:
      self._test_no_weight_skips_adversarial_loss_helper(False)

  def _test_incompatible_shapes_helper(self, main_shape, adversarial_shape):
    main_loss = tf.ones(shape=main_shape)
    adversarial_loss = tf.ones(shape=adversarial_shape)

    combined_loss = tfgan.losses.wargs.combine_adversarial_loss(
        main_loss,
        adversarial_loss,
        weight_factor=1.0,
        gradient_summaries=False)

    self.assertEqual([main_shape[0]], combined_loss.shape.as_list())

  def test_valid_incompatible_shapes(self):
    self._test_incompatible_shapes_helper(
        main_shape=[5, 4, 3, 2], adversarial_shape=[5, 3, 2])

  def test_invalid_incompatible_shapes(self):
    with self.assertRaises(ValueError):
      self._test_incompatible_shapes_helper(
          main_shape=[5, 4, 3, 2], adversarial_shape=[4, 3, 2])

  def test_stable_global_norm_avoids_overflow(self):
    tensors = [tf.ones([4]), tf.ones([4, 4]) * 1e19, None]
    gnorm_is_inf = tf.math.is_inf(tf.linalg.global_norm(tensors))
    stable_gnorm_is_inf = tf.math.is_inf(
        numerically_stable_global_norm(tensors))

    with self.cached_session() as sess:
      self.assertTrue(sess.run(gnorm_is_inf))
      self.assertFalse(sess.run(stable_gnorm_is_inf))

  def test_stable_global_norm_unchanged(self):
    """Test that preconditioning doesn't change global norm value."""
    tf.compat.v1.set_random_seed(1234)
    tensors = [tf.random.uniform([3] * i, -10.0, 10.0) for i in range(6)]
    gnorm = tf.linalg.global_norm(tensors)
    precond_gnorm = numerically_stable_global_norm(tensors)

    with self.cached_session() as sess:
      for _ in range(10):  # spot check closeness on more than one sample.
        gnorm_np, precond_gnorm_np = sess.run([gnorm, precond_gnorm])
        self.assertNear(gnorm_np, precond_gnorm_np, 1e-4)


class CycleConsistencyLossTest(tf.test.TestCase):
  """Tests for cycle_consistency_loss."""

  def setUp(self):
    super(CycleConsistencyLossTest, self).setUp()

    self._data_x_np = [[1.0, 2, 3], [4, 5, 6]]
    self._reconstructed_data_x_np = [[7.0, 8, 9], [10, 11, 12]]
    self._data_y_np = [1.0, 9]
    self._reconstructed_data_y_np = [-2.0, 3]

    self._data_x = tf.constant(self._data_x_np, dtype=tf.float32)
    self._reconstructed_data_x = tf.constant(
        self._reconstructed_data_x_np, dtype=tf.float32)
    self._data_y = tf.constant(self._data_y_np, dtype=tf.float32)
    self._reconstructed_data_y = tf.constant(
        self._reconstructed_data_y_np, dtype=tf.float32)

  def test_correct_loss(self):
    loss = tfgan.losses.wargs.cycle_consistency_loss(
        self._data_x, self._reconstructed_data_x, self._data_y,
        self._reconstructed_data_y)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertNear(5.25, sess.run(loss), 1e-5)


if __name__ == '__main__':
  tf.test.main()
