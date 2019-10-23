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

"""Tests for TF-GAN classifier_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from scipy import linalg as scp_linalg

import tensorflow as tf
import tensorflow_gan as tfgan

# Internal functions and constants to test.
from tensorflow_gan.python.eval.classifier_metrics import _classifier_score_from_logits_helper  # pylint: disable=g-bad-import-order
from tensorflow_gan.python.eval.classifier_metrics import kl_divergence
from tensorflow_gan.python.eval.classifier_metrics import trace_sqrt_product


mock = tf.compat.v1.test.mock


def _numpy_softmax(x):
  e_x = np.exp(x - np.max(x, axis=1)[:, None])
  return e_x / np.sum(e_x, axis=1)[:, None]


def _expected_inception_score(logits):
  p = _numpy_softmax(logits)
  q = np.expand_dims(np.mean(p, 0), 0)
  per_example_logincscore = np.sum(p * (np.log(p) - np.log(q)), 1)
  return np.exp(np.mean(per_example_logincscore))


def _expected_mean_only_fid(real_imgs, gen_imgs):
  m = np.mean(real_imgs, axis=0)
  m_v = np.mean(gen_imgs, axis=0)
  mean = np.square(m - m_v).sum()
  mofid = mean
  return mofid


def _expected_diagonal_only_fid(real_imgs, gen_imgs):
  m = np.mean(real_imgs, axis=0)
  m_v = np.mean(gen_imgs, axis=0)
  var = np.var(real_imgs, axis=0)
  var_v = np.var(gen_imgs, axis=0)
  sqcc = np.sqrt(var * var_v)
  mean = (np.square(m - m_v)).sum()
  trace = (var + var_v - 2 * sqcc).sum()
  dofid = mean + trace
  return dofid


def _expected_fid(real_imgs, gen_imgs):
  m = np.mean(real_imgs, axis=0)
  m_v = np.mean(gen_imgs, axis=0)
  sigma = np.cov(real_imgs, rowvar=False)
  sigma_v = np.cov(gen_imgs, rowvar=False)
  sqcc = scp_linalg.sqrtm(np.dot(sigma, sigma_v))
  mean = np.square(m - m_v).sum()
  trace = np.trace(sigma + sigma_v - 2 * sqcc)
  fid = mean + trace
  return fid


def _expected_trace_sqrt_product(sigma, sigma_v):
  return np.trace(scp_linalg.sqrtm(np.dot(sigma, sigma_v)))


def _expected_kid_and_std(real_imgs, gen_imgs, max_block_size=1024):
  n_r, dim = real_imgs.shape
  n_g = gen_imgs.shape[0]

  n_blocks = int(np.ceil(max(n_r, n_g) / max_block_size))

  sizes_r = np.full(n_blocks, n_r // n_blocks)
  to_patch = n_r - n_blocks * (n_r // n_blocks)
  if to_patch > 0:
    sizes_r[-to_patch:] += 1
  inds_r = np.r_[0, np.cumsum(sizes_r)]
  assert inds_r[-1] == n_r

  sizes_g = np.full(n_blocks, n_g // n_blocks)
  to_patch = n_g - n_blocks * (n_g // n_blocks)
  if to_patch > 0:
    sizes_g[-to_patch:] += 1
  inds_g = np.r_[0, np.cumsum(sizes_g)]
  assert inds_g[-1] == n_g

  ests = []
  for i in range(n_blocks):
    r = real_imgs[inds_r[i]:inds_r[i + 1]]
    g = gen_imgs[inds_g[i]:inds_g[i + 1]]

    k_rr = (np.dot(r, r.T) / dim + 1)**3
    k_rg = (np.dot(r, g.T) / dim + 1)**3
    k_gg = (np.dot(g, g.T) / dim + 1)**3
    ests.append(-2 * k_rg.mean() +
                k_rr[np.triu_indices_from(k_rr, k=1)].mean() +
                k_gg[np.triu_indices_from(k_gg, k=1)].mean())

  var = np.var(ests, ddof=1) if len(ests) > 1 else np.nan
  return np.mean(ests), np.sqrt(var / len(ests))


class RunClassifierFnTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RunClassifierFnTest, self).setUp()
    def multiple_outputs(x):
      return {'2x': x * 2.0, '5x': x * 5.0,
              'int': tf.ones_like(x, dtype=tf.int32)}
    def single_output(x):
      return {'2x': x * 2.0}
    self.multiple_out = multiple_outputs
    self.single_out = single_output
    self.dtypes = {'2x': tf.float32, '5x': tf.float32, 'int': tf.int32}
    self.single_dtype = {'2x': tf.float32}

  @parameterized.parameters(
      {'num_batches': 1, 'single_output': True},
      {'num_batches': 1, 'single_output': False},
      {'num_batches': 4, 'single_output': True},
      {'num_batches': 4, 'single_output': False},
  )
  def test_run_classifier_fn(self, num_batches, single_output):
    """Test graph construction."""
    img = tf.ones([8, 4, 4, 2])

    classifier_fn = self.single_out if single_output else self.multiple_out
    if single_output and num_batches == 1:
      dtypes = None
    elif single_output:
      dtypes = self.single_dtype
    else:
      dtypes = self.dtypes
    results = tfgan.eval.run_classifier_fn(
        img, classifier_fn, num_batches=num_batches, dtypes=dtypes)

    self.assertIsInstance(results, dict)
    self.assertLen(results, 1 if single_output else 3)

    self.assertIn('2x', results)
    self.assertIsInstance(results['2x'], tf.Tensor)
    self.assertAllEqual(results['2x'], img * 2)

    if not single_output:
      self.assertIn('5x', results)
      self.assertIsInstance(results['5x'], tf.Tensor)
      self.assertAllEqual(results['5x'], img * 5)

      self.assertIn('int', results)
      self.assertIsInstance(results['int'], tf.Tensor)
      self.assertAllEqual(results['int'], np.ones(results['int'].shape))

  def test_run_inception_multicall(self):
    """Test that `run_classifier_fn` can be called multiple times."""
    for batch_size in (7, 3, 2):
      img = tf.ones([batch_size, 299, 299, 3])
      tfgan.eval.run_classifier_fn(img, self.single_out)


class SampleAndRunClassifierFn(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SampleAndRunClassifierFn, self).setUp()
    def multiple_outputs(x):
      return {'2x': x * 2.0, '5x': x * 5.0,
              'int': tf.ones_like(x, dtype=tf.int32)}
    def single_output(x):
      return {'2x': x * 2.0}
    self.multiple_out = multiple_outputs
    self.single_out = single_output
    self.dtypes = {'2x': tf.float32, '5x': tf.float32, 'int': tf.int32}
    self.single_dtype = {'2x': tf.float32}

  @parameterized.parameters(
      {'num_batches': 1, 'single_output': True},
      {'num_batches': 1, 'single_output': False},
      {'num_batches': 4, 'single_output': True},
      {'num_batches': 4, 'single_output': False},
  )
  def test_sample_and_run_inception_graph(self, num_batches, single_output):
    """Test graph construction."""
    img = np.ones([8, 244, 244, 3])
    def sample_fn(_):
      return tf.constant(img, dtype=tf.float32)
    sample_inputs = [1] * num_batches

    classifier_fn = self.single_out if single_output else self.multiple_out
    if single_output and num_batches == 1:
      dtypes = None
    elif single_output:
      dtypes = self.single_dtype
    else:
      dtypes = self.dtypes

    results = tfgan.eval.sample_and_run_classifier_fn(
        sample_fn, sample_inputs, classifier_fn, dtypes=dtypes)

    self.assertIsInstance(results, dict)
    self.assertLen(results, 1 if single_output else 3)

    def _repeat(x, times):
      return np.concatenate([x] * times)

    self.assertIn('2x', results)
    self.assertIsInstance(results['2x'], tf.Tensor)
    self.assertAllEqual(results['2x'], _repeat(img * 2, num_batches))

    if not single_output:
      self.assertIn('5x', results)
      self.assertIsInstance(results['5x'], tf.Tensor)
      self.assertAllEqual(results['5x'], _repeat(img * 5, num_batches))

      self.assertIn('int', results)
      self.assertIsInstance(results['int'], tf.Tensor)
      ones = np.ones(img.shape)
      self.assertAllEqual(results['int'], _repeat(ones, num_batches))

  def test_assign_variables_in_sampler_runs(self):
    """Clarify that variables are changed by sampling function.

    This is generally an undesirable property, but rarely happens. This test is
    here to make sure that the behavior doesn't accidentally change unnoticed.
    If the sampler is ever changed to not modify the graph and this test fails,
    this test should modified or simply removed.
    """
    if tf.compat.v1.resource_variables_enabled():
      # Under the resource variables semantics the behavior of this test is
      # undefined.
      return

    def sample_fn(x):
      with tf.compat.v1.variable_scope('test', reuse=tf.compat.v1.AUTO_REUSE):
        u = tf.compat.v1.get_variable(
            'u', [1, 100],
            initializer=tf.compat.v1.truncated_normal_initializer())
        with tf.control_dependencies([u.assign(u * 2)]):
          return tf.compat.v1.layers.flatten(x * u)

    tf.compat.v1.random.set_random_seed(1023)
    sample_input = tf.random.uniform([1, 100])
    sample_inputs = [sample_input] * 10
    outputs = tfgan.eval.sample_and_run_classifier_fn(
        sample_fn, sample_inputs, self.single_out, self.single_dtype)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.global_variables())
      outputs_np = sess.run(outputs)['2x']
    self.assertEqual((10, 100), outputs_np.shape)

    for i in range(1, 10):
      self.assertFalse(np.array_equal(outputs_np[0], outputs_np[i]))


class ClassifierScoreTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ClassifierScoreTest, self).setUp()
    def classifier_fn(x):
      return 2.0 * x
    self.classifier_fn = classifier_fn

  @parameterized.parameters(
      {'num_batches': 1, 'is_streaming': False},
      {'num_batches': 4, 'is_streaming': False},
      {'num_batches': 1, 'is_streaming': True},
      {'num_batches': 4, 'is_streaming': True},
  )
  def test_classifier_score_graph(self, num_batches, is_streaming):
    """Test graph construction."""
    if is_streaming and tf.executing_eagerly():
      # Streaming is not compatible with eager execution.
      return
    input_tensor = tf.zeros([16, 32])
    fn = (tfgan.eval.classifier_score_streaming if is_streaming else
          tfgan.eval.classifier_score)
    score = fn(input_tensor, self.classifier_fn, num_batches)

    if is_streaming:
      score, update_op = score
      self.assertIsInstance(update_op, tf.Tensor)
      update_op.shape.assert_has_rank(0)
    self.assertIsInstance(score, tf.Tensor)
    score.shape.assert_has_rank(0)

  def test_classifier_score_from_logits_value(self):
    """Test value of `_classifier_score_from_logits_helper`."""
    logits = np.array(
        [np.array([1., 2.] * 500 + [4.]),
         np.array([4., 5.] * 500 + [6.])])
    unused_image = tf.zeros([2, 299, 299, 3])
    incscore = _classifier_score_from_logits_helper(logits)

    with self.cached_session(use_gpu=True) as sess:
      incscore_np = sess.run(incscore)

    self.assertAllClose(_expected_inception_score(logits), incscore_np)

  def test_streaming_classifier_score_from_logits_consistency(self):
    """Tests consistency of classifier_score_from_logits[_streaming]."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)
    num_batches = 100
    test_data = np.random.randn(num_batches, 512, 256).astype(np.float32)

    test_data_large_batch = tf.reshape(test_data, (num_batches * 512, 256))
    large_batch_score = tfgan.eval.classifier_score_from_logits(
        test_data_large_batch)

    placeholder = tf.compat.v1.placeholder(tf.float32, shape=(512, 256))
    streaming_score_value, streaming_score_update_op = (
        tfgan.eval.classifier_score_from_logits_streaming(placeholder))
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        update_op_value = sess.run(streaming_score_update_op,
                                   {placeholder: test_data[i]})
        score_value = sess.run(streaming_score_value)
        self.assertAllClose(update_op_value, score_value)
      self.assertAllClose(large_batch_score, score_value, 1e-15)


class FrechetTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(FrechetTest, self).setUp()
    def classifier_fn(x):
      return 2.0 * x
    self.classifier_fn = classifier_fn

  @parameterized.parameters(
      {'num_batches': 1, 'is_streaming': False},
      {'num_batches': 4, 'is_streaming': False},
      {'num_batches': 1, 'is_streaming': True},
      {'num_batches': 4, 'is_streaming': True},
  )
  def test_frechet_classifier_distance_graph(self, num_batches, is_streaming):
    """Test graph construction."""
    if is_streaming and tf.executing_eagerly():
      # Streaming is not compatible with eager execution.
      return
    input_tensor = tf.zeros([16, 32])
    fn = (tfgan.eval.frechet_classifier_distance_streaming if is_streaming else
          tfgan.eval.frechet_classifier_distance)
    score = fn(input_tensor, input_tensor, self.classifier_fn, num_batches)

    if is_streaming:
      score, update_op = score
      self.assertIsInstance(update_op, tf.Tensor)
      update_op.shape.assert_has_rank(0)
    self.assertIsInstance(score, tf.Tensor)
    score.shape.assert_has_rank(0)

  def test_mean_only_frechet_classifier_distance_value(self):
    """Test that `frechet_classifier_distance` gives the correct value."""
    np.random.seed(0)

    pool_real_a = np.float32(np.random.randn(256, 2048))
    pool_gen_a = np.float32(np.random.randn(256, 2048))

    tf_pool_real_a = tf.constant(pool_real_a)
    tf_pool_gen_a = tf.constant(pool_gen_a)

    mofid_op = tfgan.eval.mean_only_frechet_classifier_distance_from_activations(  # pylint: disable=line-too-long
        tf_pool_real_a, tf_pool_gen_a)

    with self.cached_session() as sess:
      actual_mofid = sess.run(mofid_op)

    expected_mofid = _expected_mean_only_fid(pool_real_a, pool_gen_a)

    self.assertAllClose(expected_mofid, actual_mofid, 0.0001)

  def test_diagonal_only_frechet_classifier_distance_value(self):
    """Test that `frechet_classifier_distance` gives the correct value."""
    np.random.seed(0)

    pool_real_a = np.float32(np.random.randn(256, 2048))
    pool_gen_a = np.float32(np.random.randn(256, 2048))

    tf_pool_real_a = tf.constant(pool_real_a)
    tf_pool_gen_a = tf.constant(pool_gen_a)

    dofid_op = tfgan.eval.diagonal_only_frechet_classifier_distance_from_activations(  # pylint: disable=line-too-long
        tf_pool_real_a, tf_pool_gen_a)

    with self.cached_session() as sess:
      actual_dofid = sess.run(dofid_op)

    expected_dofid = _expected_diagonal_only_fid(pool_real_a, pool_gen_a)

    self.assertAllClose(expected_dofid, actual_dofid, 0.0001)

  def test_frechet_classifier_distance_value(self):
    """Test that `frechet_classifier_distance` gives the correct value."""
    np.random.seed(0)

    # Make num_examples > num_features to ensure scipy's sqrtm function
    # doesn't return a complex matrix.
    test_pool_real_a = np.float32(np.random.randn(512, 256))
    test_pool_gen_a = np.float32(np.random.randn(512, 256))

    fid_op = tfgan.eval.frechet_classifier_distance(
        test_pool_real_a,
        test_pool_gen_a,
        classifier_fn=lambda x: x)

    with self.cached_session() as sess:
      actual_fid = sess.run(fid_op)

    expected_fid = _expected_fid(test_pool_real_a, test_pool_gen_a)

    self.assertAllClose(expected_fid, actual_fid, 0.0001)

  def test_streaming_frechet_classifier_distance_consistency(self):
    """Test the value of frechet_classifier_distance_streaming."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_streaming_calls = 5
    # Make num_examples > num_features to ensure scipy's sqrtm function
    # doesn't return a complex matrix.
    test_pool_real_a = np.float32(
        np.random.randn(num_streaming_calls * 512, 256))
    test_pool_gen_a = np.float32(
        np.random.randn(num_streaming_calls * 512, 256))

    real_placeholder = tf.compat.v1.placeholder(
        dtype=tf.float32, shape=(512, 256))
    gen_placeholder = tf.compat.v1.placeholder(
        dtype=tf.float32, shape=(512, 256))
    fid_value, fid_update_op = tfgan.eval.frechet_classifier_distance_streaming(
        real_placeholder,
        gen_placeholder,
        classifier_fn=lambda x: x)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_streaming_calls):
        fid_op_value = sess.run(
            fid_update_op, {
                real_placeholder: test_pool_real_a[(512 * i):(512 * (i + 1))],
                gen_placeholder: test_pool_gen_a[(512 * i):(512 * (i + 1))]
            })
        actual_fid = sess.run(fid_value)
        self.assertAllClose(fid_op_value, actual_fid)

    expected_fid = _expected_fid(test_pool_real_a, test_pool_gen_a)

    self.assertAllClose(expected_fid, actual_fid, 0.0001)

  def test_frechet_classifier_distance_covariance(self):
    """Test that `frechet_classifier_distance` takes covariance into account."""
    np.random.seed(0)

    # Make num_examples > num_features to ensure scipy's sqrtm function
    # doesn't return a complex matrix.
    test_pool_reals, test_pool_gens = [], []
    for i in range(1, 11, 2):
      test_pool_reals.append(np.float32(np.random.randn(2048, 256) * i))
      test_pool_gens.append(np.float32(np.random.randn(2048, 256) * i))

    fid_ops = []
    for i in range(len(test_pool_reals)):
      fid_ops.append(
          tfgan.eval.frechet_classifier_distance(
              test_pool_reals[i],
              test_pool_gens[i],
              classifier_fn=lambda x: x))

    fids = []
    with self.cached_session() as sess:
      for fid_op in fid_ops:
        fids.append(sess.run(fid_op))

    # Check that the FIDs increase monotonically.
    self.assertTrue(all(fid_a < fid_b for fid_a, fid_b in zip(fids, fids[1:])))


class KernelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(KernelTest, self).setUp()
    def classifier_fn(x):
      return tf.compat.v1.layers.flatten(2.0 * x)
    self.classifier_fn = classifier_fn

  @parameterized.parameters(
      {'num_batches': 1},
      {'num_batches': 4},
  )
  def test_kernel_classifier_distance_graph(self, num_batches):
    """Test `frechet_classifier_distance` graph construction."""
    input_tensor = tf.ones([8, 299, 299, 3])
    distance = tfgan.eval.kernel_classifier_distance(
        input_tensor, input_tensor, self.classifier_fn, num_batches)

    self.assertIsInstance(distance, tf.Tensor)
    distance.shape.assert_has_rank(0)

  def test_kernel_classifier_distance_value(self):
    """Test that `kernel_classifier_distance` gives the correct value."""
    np.random.seed(0)

    test_pool_real_a = np.float32(np.random.randn(512, 256))
    test_pool_gen_a = np.float32(np.random.randn(512, 256) * 1.1 + .05)

    kid_op = tfgan.eval.kernel_classifier_distance_and_std(
        test_pool_real_a,
        test_pool_gen_a,
        classifier_fn=lambda x: x,
        max_block_size=600)

    with self.cached_session() as sess:
      actual_kid, actual_std = sess.run(kid_op)

    expected_kid, expected_std = _expected_kid_and_std(test_pool_real_a,
                                                       test_pool_gen_a)

    self.assertAllClose(expected_kid, actual_kid, 0.001)
    self.assertAllClose(expected_std, actual_std, 0.001)

  def test_kernel_classifier_distance_block_sizes(self):
    """Test that function works with unusual max_block_size."""
    np.random.seed(0)

    test_pool_real_a = np.float32(np.random.randn(512, 256))
    test_pool_gen_a = np.float32(np.random.randn(768, 256) * 1.1 + .05)

    actual_expected_l = []
    if tf.executing_eagerly():
      for block_size in [50, 512, 1000]:
        actual_kid, actual_std = tfgan.eval.kernel_classifier_distance_and_std_from_activations(
            tf.constant(test_pool_real_a),
            tf.constant(test_pool_gen_a),
            max_block_size=block_size)
        expected_kid, expected_std = _expected_kid_and_std(
            test_pool_real_a, test_pool_gen_a, max_block_size=block_size)
        actual_expected_l.append((actual_kid, expected_kid))
        actual_expected_l.append((actual_std, expected_std))
    else:
      max_block_size = tf.compat.v1.placeholder(tf.int32, shape=())
      kid_op = tfgan.eval.kernel_classifier_distance_and_std_from_activations(
          tf.constant(test_pool_real_a),
          tf.constant(test_pool_gen_a),
          max_block_size=max_block_size)

      for block_size in [50, 512, 1000]:
        with self.cached_session() as sess:
          actual_kid, actual_std = sess.run(kid_op,
                                            {max_block_size: block_size})
        expected_kid, expected_std = _expected_kid_and_std(
            test_pool_real_a, test_pool_gen_a, max_block_size=block_size)
        actual_expected_l.append((actual_kid, expected_kid))
        actual_expected_l.append((actual_std, expected_std))

    for actual, expected in actual_expected_l:
      self.assertAllClose(expected, actual, 0.001)


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_trace_sqrt_product_value(self):
    """Test that `trace_sqrt_product` gives the correct value."""
    np.random.seed(0)

    # Make num_examples > num_features to ensure scipy's sqrtm function
    # doesn't return a complex matrix.
    test_pool_real_a = np.float32(np.random.randn(512, 256))
    test_pool_gen_a = np.float32(np.random.randn(512, 256))

    cov_real = np.cov(test_pool_real_a, rowvar=False)
    cov_gen = np.cov(test_pool_gen_a, rowvar=False)

    trace_sqrt_prod_op = trace_sqrt_product(cov_real, cov_gen)

    with self.cached_session() as sess:
      actual_tsp = sess.run(trace_sqrt_prod_op)

    expected_tsp = _expected_trace_sqrt_product(cov_real, cov_gen)

    self.assertAllClose(actual_tsp, expected_tsp, 0.01)

  def test_invalid_input(self):
    """Test that functions properly fail on invalid input."""
    p = tf.zeros([8, 10])
    p_logits = tf.zeros([8, 10])
    q = tf.zeros([10])
    with self.assertRaisesRegexp(ValueError, 'must be floating type'):
      kl_divergence(tf.zeros([8, 10], dtype=tf.int32), p_logits, q)

    with self.assertRaisesRegexp(ValueError, 'must be floating type'):
      kl_divergence(p, tf.zeros([8, 10], dtype=tf.int32), q)

    with self.assertRaisesRegexp(ValueError, 'must be floating type'):
      kl_divergence(p, p_logits, tf.zeros([10], dtype=tf.int32))

    with self.assertRaisesRegexp(ValueError, 'must have rank 2'):
      kl_divergence(tf.zeros([8]), p_logits, q)

    with self.assertRaisesRegexp(ValueError, 'must have rank 2'):
      kl_divergence(p, tf.zeros([8]), q)

    with self.assertRaisesRegexp(ValueError, 'must have rank 1'):
      kl_divergence(p, p_logits, tf.zeros([10, 8]))


if __name__ == '__main__':
  tf.test.main()
