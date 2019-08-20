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

"""Tests for tfgan.features.spectral_normalization."""
# TODO(tfgan): Add test that spectral normalization works with distribution
# strategies.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

import tensorflow as tf
import tensorflow_gan as tfgan


class SpectralNormalizationTest(tf.test.TestCase, parameterized.TestCase):

  def testComputeSpectralNorm(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    weights = tf.compat.v1.get_variable(
        'w', dtype=tf.float32, shape=[2, 3, 50, 100])
    weights = tf.multiply(weights, 10.0)
    s = tf.linalg.svd(
        tf.reshape(weights, [-1, weights.shape[-1]]), compute_uv=False)
    true_sn = s[..., 0]
    estimated_sn = tfgan.features.compute_spectral_norm(weights)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      np_true_sn = sess.run(true_sn)
      for i in range(50):
        est = sess.run(estimated_sn)
        if i < 1:
          np_est_1 = est
        if i < 4:
          np_est_5 = est
        if i < 9:
          np_est_10 = est
        np_est_50 = est

      # Check that the estimate improves with more iterations.
      self.assertAlmostEqual(np_true_sn, np_est_50, 0)
      self.assertGreater(
          abs(np_true_sn - np_est_10), abs(np_true_sn - np_est_50))
      self.assertGreater(
          abs(np_true_sn - np_est_5), abs(np_true_sn - np_est_10))
      self.assertGreater(abs(np_true_sn - np_est_1), abs(np_true_sn - np_est_5))

  def testSpectralNormalize(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    weights = tf.compat.v1.get_variable(
        'w', dtype=tf.float32, shape=[2, 3, 50, 100])
    weights = tf.multiply(weights, 10.0)
    normalized_weights = tfgan.features.spectral_normalize(
        weights, power_iteration_rounds=1)

    unnormalized_sigma = tf.linalg.svd(
        tf.reshape(weights, [-1, weights.shape[-1]]), compute_uv=False)[..., 0]
    normalized_sigma = tf.linalg.svd(
        tf.reshape(normalized_weights, [-1, weights.shape[-1]]),
        compute_uv=False)[..., 0]

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      s0 = sess.run(unnormalized_sigma)

      for i in range(50):
        sigma = sess.run(normalized_sigma)
        if i < 1:
          s1 = sigma
        if i < 5:
          s5 = sigma
        if i < 10:
          s10 = sigma
        s50 = sigma

      self.assertAlmostEqual(1., s50, 0)
      self.assertGreater(abs(s10 - 1.), abs(s50 - 1.))
      self.assertGreater(abs(s5 - 1.), abs(s10 - 1.))
      self.assertGreater(abs(s1 - 1.), abs(s5 - 1.))
      self.assertGreater(abs(s0 - 1.), abs(s1 - 1.))

  def testSpectralNormalizeZeroMatrix(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    w = tf.zeros(shape=[2, 3, 50, 100])
    normalized_w = tfgan.features.spectral_normalize(
        w, power_iteration_rounds=5, equality_constrained=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      np_normalized_w = sess.run(normalized_w)

    normalized_w_norm = np.linalg.svd(np_normalized_w.reshape([-1, 3]))[1][0]
    self.assertAllClose(normalized_w_norm, 0.)

  def testSpectralNormalizeTinyMatrix(self):
    """Test spectral_normalize when normalization_threshold is None."""
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    w = tf.ones(shape=[2, 3, 50, 100]) * 1e-5
    normalized_w = tfgan.features.spectral_normalize(
        w, power_iteration_rounds=5, equality_constrained=True)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      np_normalized_w = sess.run(normalized_w)
    normalized_w_norm = np.linalg.svd(np_normalized_w.reshape([-1, 3]))[1][0]
    self.assertAllClose(normalized_w_norm, 1., rtol=1e-4, atol=1e-4)

  def _testLayerHelper(self, build_layer_fn, w_shape, b_shape):
    x = tf.zeros([2, 10, 10, 3], dtype=tf.float32)

    w_initial = np.random.randn(*w_shape) * 10
    w_initializer = tf.compat.v1.initializers.constant(w_initial)
    b_initial = np.random.randn(*b_shape)
    b_initializer = tf.compat.v1.initializers.constant(b_initial)

    getter = tfgan.features.spectral_normalization_custom_getter()
    context_manager = tf.compat.v1.variable_scope('', custom_getter=getter)

    with context_manager:
      (net,
       expected_normalized_vars, expected_not_normalized_vars) = build_layer_fn(
           x, w_initializer, b_initializer)

    x_data = np.random.rand(*x.shape)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())

      # Before running a forward pass we still expect the variables values to
      # differ from the initial value because of the normalizer.
      w_befores = []
      for name, var in expected_normalized_vars.items():
        w_before = sess.run(var)
        w_befores.append(w_before)
        self.assertFalse(
            np.allclose(w_initial, w_before),
            msg=('%s appears not to be normalized. Before: %s After: %s' %
                 (name, w_initial, w_before)))

      # Not true for the unnormalized variables.
      for name, var in expected_not_normalized_vars.items():
        b_before = sess.run(var)
        self.assertTrue(
            np.allclose(b_initial, b_before),
            msg=('%s appears to be unexpectedly normalized. '
                 'Before: %s After: %s' % (name, b_initial, b_before)))

      # Run a bunch of forward passes.
      for _ in range(1000):
        _ = sess.run(net, feed_dict={x: x_data})

      # We expect this to have improved the estimate of the spectral norm,
      # which should have changed the variable values and brought them close
      # to the true Spectral Normalized values.
      _, s, _ = np.linalg.svd(w_initial.reshape([-1, 3]))
      exactly_normalized = w_initial / s[0]
      for w_before, (name, var) in zip(w_befores,
                                       expected_normalized_vars.items()):
        w_after = sess.run(var)
        self.assertFalse(
            np.allclose(w_before, w_after, rtol=1e-8, atol=1e-8),
            msg=('%s did not improve over many iterations. '
                 'Before: %s After: %s' % (name, w_before, w_after)))
        self.assertAllClose(
            exactly_normalized,
            w_after,
            rtol=1e-4,
            atol=1e-4,
            msg=('Estimate of spectral norm for %s was innacurate. '
                 'Normalized matrices do not match.'
                 'Estimate: %s Actual: %s' % (name, w_after,
                                              exactly_normalized)))

  def testConv2D_Layers(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return

    def build_layer_fn(x, w_initializer, b_initializer):
      layer = tf.compat.v1.layers.Conv2D(
          filters=3,
          kernel_size=3,
          padding='same',
          kernel_initializer=w_initializer,
          bias_initializer=b_initializer)
      net = layer.apply(x)
      expected_normalized_vars = {'tf.layers.Conv2d.kernel': layer.kernel}
      expected_not_normalized_vars = {'tf.layers.Conv2d.bias': layer.bias}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (3, 3, 3, 3), (3,))

  def testConv2D_Keras(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return

    def build_layer_fn(x, w_initializer, b_initializer):
      layer = tf.keras.layers.Conv2D(
          filters=3,
          kernel_size=3,
          padding='same',
          kernel_initializer=w_initializer,
          bias_initializer=b_initializer)
      layer.build(x.shape)
      layer.kernel = tfgan.features.spectral_normalize(layer.kernel)
      net = layer.apply(x)
      expected_normalized_vars = {'keras.Conv2d.kernel': layer.kernel}
      expected_not_normalized_vars = {'keras.Conv2d.bias': layer.bias}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (3, 3, 3, 3), (3,))

  def testConv2D_ContribLayers(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return

    try:
      tf.contrib.layers.conv2d
    except AttributeError:  # if contrib doesn't exist, skip this test.
      return True

    def build_layer_fn(x, w_initializer, b_initializer):
      var_collection = {
          'weights': ['CONTRIB_LAYERS_CONV2D_WEIGHTS'],
          'biases': ['CONTRIB_LAYERS_CONV2D_BIASES']
      }
      net = tf.contrib.layers.conv2d(
          x,
          3,
          3,
          weights_initializer=w_initializer,
          biases_initializer=b_initializer,
          variables_collections=var_collection)
      weight_vars = tf.compat.v1.get_collection('CONTRIB_LAYERS_CONV2D_WEIGHTS')
      self.assertLen(weight_vars, 1)
      bias_vars = tf.compat.v1.get_collection('CONTRIB_LAYERS_CONV2D_BIASES')
      self.assertLen(bias_vars, 1)
      expected_normalized_vars = {
          'contrib.layers.conv2d.weights': weight_vars[0]
      }
      expected_not_normalized_vars = {
          'contrib.layers.conv2d.bias': bias_vars[0]
      }

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (3, 3, 3, 3), (3,))

  def testConv2D_Slim(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return

    try:
      tf.contrib.slim.conv2d
    except AttributeError:  # if contrib doesn't exist, skip this test.
      return True

    def build_layer_fn(x, w_initializer, b_initializer):
      var_collection = {
          'weights': ['SLIM_CONV2D_WEIGHTS'],
          'biases': ['SLIM_CONV2D_BIASES']
      }
      net = tf.contrib.slim.conv2d(
          x,
          3,
          3,
          weights_initializer=w_initializer,
          biases_initializer=b_initializer,
          variables_collections=var_collection)
      weight_vars = tf.compat.v1.get_collection('SLIM_CONV2D_WEIGHTS')
      self.assertLen(weight_vars, 1)
      bias_vars = tf.compat.v1.get_collection('SLIM_CONV2D_BIASES')
      self.assertLen(bias_vars, 1)
      expected_normalized_vars = {'slim.conv2d.weights': weight_vars[0]}
      expected_not_normalized_vars = {'slim.conv2d.bias': bias_vars[0]}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (3, 3, 3, 3), (3,))

  def testFC_Layers(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return

    def build_layer_fn(x, w_initializer, b_initializer):
      x = tf.compat.v1.layers.flatten(x)
      layer = tf.compat.v1.layers.Dense(
          units=3,
          kernel_initializer=w_initializer,
          bias_initializer=b_initializer)
      net = layer.apply(x)
      expected_normalized_vars = {'tf.layers.Dense.kernel': layer.kernel}
      expected_not_normalized_vars = {'tf.layers.Dense.bias': layer.bias}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (300, 3), (3,))

  def testFC_Keras(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return

    def build_layer_fn(x, w_initializer, b_initializer):
      flatten = tf.keras.layers.Flatten()
      x = flatten.apply(x)
      layer = tf.keras.layers.Dense(
          units=3,
          kernel_initializer=w_initializer,
          bias_initializer=b_initializer)
      layer.build(x.shape)
      layer.kernel = tfgan.features.spectral_normalize(layer.kernel)
      net = layer.apply(x)
      expected_normalized_vars = {'keras.Dense.kernel': layer.kernel}
      expected_not_normalized_vars = {'keras.Dense.bias': layer.bias}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (300, 3), (3,))

  def testFC_ContribLayers(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    try:
      tf.contrib.layers.fully_connected
    except AttributeError:  # if contrib doesn't exist, skip this test.
      return

    def build_layer_fn(x, w_initializer, b_initializer):
      var_collection = {
          'weights': ['CONTRIB_LAYERS_FC_WEIGHTS'],
          'biases': ['CONTRIB_LAYERS_FC_BIASES']
      }
      x = tf.contrib.layers.flatten(x)
      net = tf.contrib.layers.fully_connected(
          x,
          3,
          weights_initializer=w_initializer,
          biases_initializer=b_initializer,
          variables_collections=var_collection)
      weight_vars = tf.compat.v1.get_collection('CONTRIB_LAYERS_FC_WEIGHTS')
      self.assertLen(weight_vars, 1)
      bias_vars = tf.compat.v1.get_collection('CONTRIB_LAYERS_FC_BIASES')
      self.assertLen(bias_vars, 1)
      expected_normalized_vars = {
          'contrib.layers.fully_connected.weights': weight_vars[0]
      }
      expected_not_normalized_vars = {
          'contrib.layers.fully_connected.bias': bias_vars[0]
      }

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (300, 3), (3,))

  def testFC_Slim(self):
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return

    try:
      tf.contrib.slim.fully_connected
    except AttributeError:  # if contrib doesn't exist, skip this test.
      return

    def build_layer_fn(x, w_initializer, b_initializer):
      var_collection = {
          'weights': ['SLIM_FC_WEIGHTS'],
          'biases': ['SLIM_FC_BIASES']
      }
      x = tf.contrib.slim.flatten(x)
      net = tf.contrib.slim.fully_connected(
          x,
          3,
          weights_initializer=w_initializer,
          biases_initializer=b_initializer,
          variables_collections=var_collection)
      weight_vars = tf.compat.v1.get_collection('SLIM_FC_WEIGHTS')
      self.assertLen(weight_vars, 1)
      bias_vars = tf.compat.v1.get_collection('SLIM_FC_BIASES')
      self.assertLen(bias_vars, 1)
      expected_normalized_vars = {
          'slim.fully_connected.weights': weight_vars[0]
      }
      expected_not_normalized_vars = {'slim.fully_connected.bias': bias_vars[0]}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (300, 3), (3,))

  @parameterized.parameters(
      {'repeat_type': 'with_dep', 'training': True, 'expect_same': False},
      {'repeat_type': 'double', 'training': False, 'expect_same': True},
      {'repeat_type': 'with_dep', 'training': False, 'expect_same': True},
      {'repeat_type': 'map_fn', 'training': False, 'expect_same': True},
  )
  def test_multiple_calls(self, repeat_type, training, expect_same):
    """Tests that multiple calls don't change variables."""
    if tf.executing_eagerly():
      # `compute_spectral_norm` doesn't work when executing eagerly.
      return
    sn_gettr = tfgan.features.spectral_normalization_custom_getter
    output_size = 100
    def generator(x):
      with tf.compat.v1.variable_scope(
          'gen',
          custom_getter=sn_gettr(training=training),
          reuse=tf.compat.v1.AUTO_REUSE):
        return tf.compat.v1.layers.dense(
            x,
            units=output_size,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(),
            bias_initializer=tf.compat.v1.truncated_normal_initializer())

    i = tf.random.uniform([1, 10])
    if repeat_type == 'double':
      output1 = generator(i)
      output2 = generator(i)
    elif repeat_type == 'with_dep':
      output1 = generator(i)
      with tf.control_dependencies([output1]):
        output2 = generator(i)
    else:  # map_fn
      num_loops = 2
      input_list = [i] * num_loops
      outputs = tf.map_fn(
          generator,
          tf.stack(input_list),
          parallel_iterations=1,
          back_prop=False,
          swap_memory=True)
      outputs.shape.assert_is_compatible_with((num_loops, 1, output_size))
      output1 = tf.expand_dims(outputs[0, 0, :], 0)
      output2 = tf.expand_dims(outputs[1, 0, :], 0)

    with tf.compat.v1.train.MonitoredSession() as sess:
      o1, o2 = sess.run([output1, output2])
    self.assertAllEqual((1, output_size), o1.shape)
    self.assertAllEqual((1, output_size), o2.shape)

    if expect_same:
      self.assertAllEqual(o1, o2)
    else:
      self.assertFalse(np.array_equal(o1, o2))

if __name__ == '__main__':
  tf.test.main()
