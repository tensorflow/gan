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

"""Tests for VariableClippingOptimizer.

This entire thing is copied from `third_party/tensorflow/contrib/opt/python/
training/variable_clipping_optimizer_test.py`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import socket

import numpy as np

import tensorflow as tf
import tensorflow_gan as tfgan


class VariableClippingOptimizerTest(tf.test.TestCase):

  def _setupCluster(self):

    def get_open_port():
      try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      except IOError:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
      s.bind(("", 0))
      port = s.getsockname()[1]
      s.close()
      return port

    port1 = get_open_port()
    port2 = get_open_port()
    cs = tf.train.ClusterSpec({
        "worker": ["localhost:%s" % port1],
        "ps": ["localhost:%s" % port2]
    })

    worker = tf.distribute.Server(cs, job_name="worker", start=True)
    ps = tf.distribute.Server(cs, job_name="ps", start=True)

    return worker, ps

  @contextlib.contextmanager
  def _maybeWithDevice(self, device):
    if device is not None:
      with tf.device(device):
        yield
    else:
      yield

  def _setupDense(self, is_distributed, dtype, sess):
    with self._maybeWithDevice("/job:ps" if is_distributed else None):
      var0 = tf.Variable([[0.0, 1.0], [2.0, 3.0]], dtype=dtype)
      var1 = tf.Variable([4.0, 5.0], dtype=dtype)
    with self._maybeWithDevice("/job:worker" if is_distributed else None):
      grads0 = tf.constant([[0.1, 0.1], [0.1, 0.1]], dtype=dtype)
      grads1 = tf.constant([0.01, 0.01], dtype=dtype)
      sgd = tf.compat.v1.train.GradientDescentOptimizer(3.0)
      clip_opt = tfgan.features.VariableClippingOptimizer(sgd, {var0: [1]}, 2.0)

      update_op = clip_opt.apply_gradients(
          list(zip([grads0, grads1], [var0, var1])))
      sess.run(tf.compat.v1.global_variables_initializer())
    return var0, var1, update_op

  def _assertDenseCorrect(self, var0, var1, update_op, sess):
    # Fetch params to validate initial values
    self.assertAllCloseAccordingToType([[0.0, 1.0], [2.0, 3.0]], sess.run(var0))
    self.assertAllCloseAccordingToType([4.0, 5.0], sess.run(var1))

    # Run 1 step of sgd, clipping each var0[i] to max L2-norm 2.0
    sess.run(update_op)
    # Validate updated params
    var0_out = sess.run(var0)
    # var0[0] has norm < 2.0, so it is not clipped.
    self.assertAllCloseAccordingToType([(0.0 - 3.0 * 0.1), (1.0 - 3.0 * 0.1)],
                                       var0_out[0])
    # var0[1] has norm > 2.0, so it is clipped.
    expected_unclipped = np.array([(2.0 - 3.0 * 0.1), (3.0 - 3.0 * 0.1)])
    self.assertAllCloseAccordingToType(
        2.0 * expected_unclipped / np.linalg.norm(expected_unclipped),
        var0_out[1])
    # var1 is not in the var list, so it should not be clipped
    self.assertAllCloseAccordingToType([4.0 - 3.0 * 0.01, 5.0 - 3.0 * 0.01],
                                       sess.run(var1))

  def _setupSparse(self, is_distributed, dtype, sess):
    with self._maybeWithDevice("/job:ps" if is_distributed else None):
      var0 = tf.Variable([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=dtype)
      var1 = tf.Variable([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]], dtype=dtype)
    with self._maybeWithDevice("/job:worker" if is_distributed else None):
      grads = tf.IndexedSlices(
          tf.constant([[0.1, 0.1], [0.1, 0.1]], dtype=dtype), [0, 2], [3, 2])
      sgd = tf.compat.v1.train.GradientDescentOptimizer(3.0)
      clip_opt = tfgan.features.VariableClippingOptimizer(
          sgd, {
              var0: [1],
              var1: [0]
          }, 2.0)
      update_op = clip_opt.apply_gradients(
          list(zip([grads, grads], [var0, var1])))
      sess.run(tf.compat.v1.global_variables_initializer())
    return var0, var1, update_op

  def _assertSparseCorrect(self, var0, var1, update_op, sess):
    # Fetch params to validate initial values
    self.assertAllCloseAccordingToType([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                                       sess.run(var0))
    self.assertAllCloseAccordingToType([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]],
                                       sess.run(var1))

    # Run 1 step of sgd
    sess.run(update_op)

    # var1 is clipped along the sparse dimension, so defaults to using dense
    # calculations. There should be a warning logged, but the numerics
    # should still be correct.
    var1_out = sess.run(var1)
    # var1[:, 0] has norm < 2.0, so it is not clipped.
    self.assertAllCloseAccordingToType([(0.0 - 3.0 * 0.1), 0.0,
                                        (0.0 - 3.0 * 0.1)], var1_out[:, 0])
    # var1[:, 1] has norm > 2.0, so it is clipped.
    expected_unclipped = np.array([(1.0 - 3.0 * 0.1), 3.0, (5.0 - 3.0 * 0.1)])
    self.assertAllCloseAccordingToType(
        2.0 * expected_unclipped / np.linalg.norm(expected_unclipped),
        var1_out[:, 1])

    # Validate updated params
    var0_out = sess.run(var0)
    # var0[0] has norm < 2.0, so it is not clipped.
    self.assertAllCloseAccordingToType([(0.0 - 3.0 * 0.1), (1.0 - 3.0 * 0.1)],
                                       var0_out[0])
    # var0[1] has no gradients, so it should remain unchanged.
    self.assertAllCloseAccordingToType([2.0, 3.0], var0_out[1])
    # var0[2] has norm > 2.0, so it is clipped.
    expected_unclipped = np.array([(4.0 - 3.0 * 0.1), (5.0 - 3.0 * 0.1)])
    self.assertAllCloseAccordingToType(
        2.0 * expected_unclipped / np.linalg.norm(expected_unclipped),
        var0_out[2])

  def testDenseLocal(self):
    if tf.executing_eagerly():
      # Test not designed to run in eager mode.
      return
    for dtype in [tf.float32, tf.float64, tf.half]:
      with self.cached_session() as sess:
        var0, var1, update_op = self._setupDense(False, dtype, sess)
        self._assertDenseCorrect(var0, var1, update_op, sess)

  def testDenseDistributed(self):
    worker, unused_ps = self._setupCluster()
    for dtype in [tf.float64, tf.half, tf.float32]:
      with tf.compat.v1.Session(worker.target) as sess:
        var0, var1, update_op = self._setupDense(True, dtype, sess)
        self._assertDenseCorrect(var0, var1, update_op, sess)

  def testSparseLocal(self):
    if tf.executing_eagerly():
      # Test not designed to run in eager mode.
      return
    for dtype in [tf.float64, tf.float32, tf.half]:
      with self.cached_session() as sess:
        var0, var1, update_op = self._setupSparse(False, dtype, sess)
        self._assertSparseCorrect(var0, var1, update_op, sess)

  def testSparseDistributed(self):
    worker, unused_ps = self._setupCluster()
    for dtype in [tf.half, tf.float32, tf.float64]:
      with tf.compat.v1.Session(worker.target) as sess:
        var0, var1, update_op = self._setupSparse(True, dtype, sess)
        self._assertSparseCorrect(var0, var1, update_op, sess)


if __name__ == "__main__":
  tf.test.main()
