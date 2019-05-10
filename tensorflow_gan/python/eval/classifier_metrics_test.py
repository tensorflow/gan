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

"""Tests for TF-GAN tfgan.eval.classifier_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import tempfile

from absl.testing import parameterized
import numpy as np
from scipy import linalg as scp_linalg

import tensorflow as tf
import tensorflow_gan as tfgan

# Internal functions to mock.
from tensorflow_gan.python.eval.classifier_metrics import get_graph_def_from_url_tarball

# Internal functions and constants to test.
from tensorflow_gan.python.eval.classifier_metrics import INCEPTION_FINAL_POOL
from tensorflow_gan.python.eval.classifier_metrics import INCEPTION_OUTPUT
from tensorflow_gan.python.eval.classifier_metrics import kl_divergence
from tensorflow_gan.python.eval.classifier_metrics import trace_sqrt_product

from google.protobuf import text_format
from tensorflow.python import tf2  # pylint:disable=g-direct-tensorflow-import

mock = tf.compat.v1.test.mock
classifier_module = sys.modules[get_graph_def_from_url_tarball.__module__]


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


# A dummy GraphDef string with the minimum number of Ops.
graphdef_string = """
node {
  name: "Mul"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 299
        }
        dim {
          size: 299
        }
        dim {
          size: 3
        }
      }
    }
  }
}
node {
  name: "logits"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 1001
        }
      }
    }
  }
}
node {
  name: "pool_3"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 2048
        }
      }
    }
  }
}
versions {
  producer: 24
}
"""


def _get_dummy_graphdef():
  dummy_graphdef = tf.compat.v1.GraphDef()
  text_format.Merge(graphdef_string, dummy_graphdef)
  return dummy_graphdef


def _run_with_mock(function, *args, **kwargs):
  with mock.patch.object(
      classifier_module,
      'get_graph_def_from_url_tarball') as mock_tarball_getter:
    mock_tarball_getter.return_value = _get_dummy_graphdef()
    return function(*args, **kwargs)


class RunInceptionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'use_default_graph_def': False, 'singleton': True, 'num_batches': 1},
      {'use_default_graph_def': False, 'singleton': True, 'num_batches': 4},
      {'use_default_graph_def': True, 'singleton': True, 'num_batches': 1},
      {'use_default_graph_def': True, 'singleton': True, 'num_batches': 4},
      {'use_default_graph_def': False, 'singleton': False, 'num_batches': 1},
  )
  def test_run_inception_graph(self, use_default_graph_def, singleton,
                               num_batches):
    """Test `run_inception` graph construction."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return

    batch_size = 8
    img = tf.ones([batch_size, 299, 299, 3])
    output_tensor = INCEPTION_OUTPUT if singleton else [INCEPTION_OUTPUT]

    if use_default_graph_def:
      logits = _run_with_mock(
          tfgan.eval.run_inception, img, output_tensor=output_tensor,
          num_batches=num_batches)
    else:
      logits = tfgan.eval.run_inception(
          img, _get_dummy_graphdef(), output_tensor=output_tensor,
          num_batches=num_batches)

    if not singleton:
      self.assertIsInstance(logits, list)
      logits = logits[0]
    self.assertIsInstance(logits, tf.Tensor)
    logits.shape.assert_is_compatible_with([batch_size, 1001])

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())

  @parameterized.parameters(
      {'use_default_graph_def': True, 'num_batches': 1},
      {'use_default_graph_def': True, 'num_batches': 4},
      {'use_default_graph_def': False, 'num_batches': 1},
      {'use_default_graph_def': False, 'num_batches': 4},
  )
  def test_run_inception_graph_pool_output(self, use_default_graph_def,
                                           num_batches):
    """Test `run_inception` graph construction with pool output."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return

    batch_size = 8
    img = tf.ones([batch_size, 299, 299, 3])

    if use_default_graph_def:
      pool = _run_with_mock(
          tfgan.eval.run_inception, img, output_tensor=INCEPTION_FINAL_POOL,
          num_batches=num_batches)
    else:
      pool = tfgan.eval.run_inception(
          img, _get_dummy_graphdef(), output_tensor=INCEPTION_FINAL_POOL,
          num_batches=num_batches)

    self.assertIsInstance(pool, tf.Tensor)
    pool.shape.assert_is_compatible_with([batch_size, 2048])

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())

  def test_run_inception_unicode(self):
    """Test `run_inception` with unicode input and output names."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return

    batch_size = 8
    img = tf.ones([batch_size, 299, 299, 3])

    tfgan.eval.run_inception(
        img, _get_dummy_graphdef(), output_tensor=u'pool_3:0', num_batches=2)

  def test_run_inception_multiple_outputs(self):
    """Test `run_inception` graph construction with multiple outputs."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return

    batch_size = 3
    img = tf.ones([batch_size, 299, 299, 3])
    logits, pool = _run_with_mock(
        tfgan.eval.run_inception,
        img,
        output_tensor=[INCEPTION_OUTPUT, INCEPTION_FINAL_POOL])

    self.assertIsInstance(logits, tf.Tensor)
    self.assertIsInstance(pool, tf.Tensor)
    logits.shape.assert_is_compatible_with([batch_size, 1001])
    pool.shape.assert_is_compatible_with([batch_size, 2048])

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())

  def test_run_inception_multicall(self):
    """Test that `run_inception` can be called multiple times."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return

    for batch_size in (7, 3, 2):
      img = tf.ones([batch_size, 299, 299, 3])
      _run_with_mock(tfgan.eval.run_inception, img)

  def test_invalid_input(self):
    """Test that functions properly fail on invalid input."""
    with self.assertRaisesRegexp(ValueError, 'Shapes .* are incompatible'):
      tfgan.eval.run_inception(tf.ones([7, 50, 50, 3]))

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


class InceptionScoreTest(tf.test.TestCase, parameterized.TestCase):

  def test_inception_score_graph(self):
    """Test `inception_score` graph construction."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return

    score = _run_with_mock(
        tfgan.eval.inception_score, tf.zeros([6, 299, 299, 3]), num_batches=3)
    self.assertIsInstance(score, tf.Tensor)
    score.shape.assert_has_rank(0)

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())

  def test_inception_score_value(self):
    """Test that `inception_score` gives the correct value."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return

    logits = np.array(
        [np.array([1, 2] * 500 + [4]),
         np.array([4, 5] * 500 + [6])])
    unused_image = tf.zeros([2, 299, 299, 3])
    incscore = _run_with_mock(tfgan.eval.inception_score, unused_image)

    with self.cached_session(use_gpu=True) as sess:
      incscore_np = sess.run(incscore, {'concat:0': logits})

    self.assertAllClose(_expected_inception_score(logits), incscore_np)


class SampleAndClassifyTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'use_default_graph_def': False, 'num_outputs': 0, 'num_batches': 1},
      {'use_default_graph_def': False, 'num_outputs': 0, 'num_batches': 4},
      {'use_default_graph_def': False, 'num_outputs': 1, 'num_batches': 1},
      {'use_default_graph_def': False, 'num_outputs': 1, 'num_batches': 4},
      {'use_default_graph_def': False, 'num_outputs': 2, 'num_batches': 1},
      {'use_default_graph_def': False, 'num_outputs': 2, 'num_batches': 4},
      {'use_default_graph_def': True, 'num_outputs': 0, 'num_batches': 1},
      {'use_default_graph_def': True, 'num_outputs': 0, 'num_batches': 4},
      {'use_default_graph_def': True, 'num_outputs': 1, 'num_batches': 1},
      {'use_default_graph_def': True, 'num_outputs': 1, 'num_batches': 4},
      {'use_default_graph_def': True, 'num_outputs': 2, 'num_batches': 1},
      {'use_default_graph_def': True, 'num_outputs': 2, 'num_batches': 4},
  )
  def test_sample_and_run_inception_graph(
      self, use_default_graph_def, num_outputs, num_batches):
    """Test `test_sample_and_run_inception_graph` graph construction."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return
    batch_size = 8
    def sample_fn(_):
      return tf.ones([batch_size, 299, 299, 3])
    sample_inputs = [1] * num_batches

    output_tensor = {
        0: INCEPTION_OUTPUT,
        1: [INCEPTION_OUTPUT],
        2: [INCEPTION_OUTPUT, INCEPTION_FINAL_POOL],
    }[num_outputs]

    if use_default_graph_def:
      logits = _run_with_mock(
          tfgan.eval.sample_and_run_inception, sample_fn, sample_inputs,
          output_tensor=output_tensor)
    else:
      logits = tfgan.eval.sample_and_run_inception(
          sample_fn, sample_inputs, _get_dummy_graphdef(),
          output_tensor=output_tensor)

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())

    if num_outputs == 0:
      self.assertIsInstance(logits, tf.Tensor)
      logits.shape.assert_is_compatible_with([batch_size, 1001])
    elif num_outputs == 1:
      self.assertIsInstance(logits, list)
      self.assertLen(logits, 1)
      logits[0].shape.assert_is_compatible_with([batch_size, 1001])
    elif num_outputs == 2:
      self.assertIsInstance(logits, list)
      self.assertLen(logits, 2)
      logits[0].shape.assert_is_compatible_with([batch_size, 1001])
      logits[1].shape.assert_is_compatible_with([batch_size, 2048])

  def test_assign_variables_in_sampler_runs(self):
    """Clarify that variables are changed by sampling function.

    This is generally an undesirable property, but rarely happens. This test is
    here to make sure that the behavior doesn't accidentally change unnoticed.
    If the sampler is ever changed to not modify the graph and this test fails,
    this test should modified or simply removed.
    """
    if tf2.enabled():
      # `run_image_classifier` doesn't work in eager execution.
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
    g = tf.Graph()
    with g.as_default():
      input_tensor = tf.compat.v1.placeholder(
          tf.float32, shape=[None, 1], name='input')
      output = input_tensor * 2
    outputs = tfgan.eval.sample_and_run_image_classifier(
        sample_fn, sample_inputs, g.as_graph_def(), input_tensor.name,
        output.name)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.global_variables())
      outputs_np = sess.run(outputs)
    self.assertEqual((10, 100), outputs_np.shape)

    for i in range(1, 10):
      self.assertFalse(np.array_equal(outputs_np[0], outputs_np[i]))


class FIDTest(tf.test.TestCase, parameterized.TestCase):

  def test_frechet_inception_distance_graph(self):
    """Test `frechet_inception_distance` graph construction."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return
    img = tf.ones([7, 299, 299, 3])
    distance = _run_with_mock(tfgan.eval.frechet_inception_distance, img, img)

    self.assertIsInstance(distance, tf.Tensor)
    distance.shape.assert_has_rank(0)

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())

  def test_kernel_inception_distance_graph(self):
    """Test `frechet_inception_distance` graph construction."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return
    img = tf.ones([7, 299, 299, 3])
    distance = _run_with_mock(tfgan.eval.kernel_inception_distance, img, img)

    self.assertIsInstance(distance, tf.Tensor)
    distance.shape.assert_has_rank(0)

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())

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

    fid_op = _run_with_mock(
        tfgan.eval.frechet_classifier_distance,
        test_pool_real_a,
        test_pool_gen_a,
        classifier_fn=lambda x: x)

    with self.cached_session() as sess:
      actual_fid = sess.run(fid_op)

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
          _run_with_mock(
              tfgan.eval.frechet_classifier_distance,
              test_pool_reals[i],
              test_pool_gens[i],
              classifier_fn=lambda x: x))

    fids = []
    with self.cached_session() as sess:
      for fid_op in fid_ops:
        fids.append(sess.run(fid_op))

    # Check that the FIDs increase monotonically.
    self.assertTrue(all(fid_a < fid_b for fid_a, fid_b in zip(fids, fids[1:])))

  def test_kernel_classifier_distance_value(self):
    """Test that `kernel_classifier_distance` gives the correct value."""
    np.random.seed(0)

    test_pool_real_a = np.float32(np.random.randn(512, 256))
    test_pool_gen_a = np.float32(np.random.randn(512, 256) * 1.1 + .05)

    kid_op = _run_with_mock(
        tfgan.eval.kernel_classifier_distance_and_std,
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
        actual_kid, actual_std = _run_with_mock(
            tfgan.eval.kernel_classifier_distance_and_std_from_activations,
            tf.constant(test_pool_real_a),
            tf.constant(test_pool_gen_a),
            max_block_size=block_size)
        expected_kid, expected_std = _expected_kid_and_std(
            test_pool_real_a, test_pool_gen_a, max_block_size=block_size)
        actual_expected_l.append((actual_kid, expected_kid))
        actual_expected_l.append((actual_std, expected_std))
    else:
      max_block_size = tf.compat.v1.placeholder(tf.int32, shape=())
      kid_op = _run_with_mock(
          tfgan.eval.kernel_classifier_distance_and_std_from_activations,
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

    trace_sqrt_prod_op = _run_with_mock(trace_sqrt_product, cov_real, cov_gen)

    with self.cached_session() as sess:
      # trace_sqrt_product: tsp
      actual_tsp = sess.run(trace_sqrt_prod_op)

    expected_tsp = _expected_trace_sqrt_product(cov_real, cov_gen)

    self.assertAllClose(actual_tsp, expected_tsp, 0.01)

  def test_preprocess_image_graph(self):
    """Test `preprocess_image` graph construction."""
    if tf.executing_eagerly():
      # `run_image_classifier` doesn't work in eager execution.
      return
    incorrectly_sized_image = tf.zeros([520, 240, 3])
    correct_image = tfgan.eval.preprocess_image(images=incorrectly_sized_image)
    _run_with_mock(tfgan.eval.run_inception, tf.expand_dims(correct_image, 0))

  def test_get_graph_def_from_url_tarball(self):
    """Test `get_graph_def_from_url_tarball`."""
    # Write dummy binary GraphDef to tempfile.
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      tmp_file.write(_get_dummy_graphdef().SerializeToString())
    relative_path = os.path.relpath(tmp_file.name)

    # Create gzip tarball.
    tar_dir = tempfile.mkdtemp()
    tar_filename = os.path.join(tar_dir, 'tmp.tar.gz')
    with tarfile.open(tar_filename, 'w:gz') as tar:
      tar.add(relative_path)

    with mock.patch.object(classifier_module, 'urllib') as mock_urllib:
      mock_urllib.request.urlretrieve.return_value = tar_filename, None
      graph_def = tfgan.eval.get_graph_def_from_url_tarball(
          'unused_url', relative_path)

    self.assertIsInstance(graph_def, tf.compat.v1.GraphDef)
    self.assertEqual(_get_dummy_graphdef(), graph_def)


if __name__ == '__main__':
  tf.test.main()
