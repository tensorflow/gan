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

"""Tests for TF-GAN internal inception_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_gan as tfgan

mock = tf.compat.v1.test.mock


class FakeInceptionModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def __call__(self, x):
    bs = tf.shape(x)[0]
    logits = tf.zeros([bs, 1008])
    pool_3 = tf.ones([bs, 2048])
    return {'logits': logits, 'pool_3': pool_3}


class RunInceptionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RunInceptionTest, self).setUp()
    self.export_path = os.path.join(self.get_temp_dir(), 'my-module')
    tf.saved_model.save(FakeInceptionModule(), self.export_path)
    classifier_fn = tfgan.eval.classifier_fn_from_tfhub(
        self.export_path, None)
    def run_inception(*args, **kwargs):
      return tfgan.eval.run_inception(
          *args, classifier_fn=classifier_fn, **kwargs)
    self.run_inception = run_inception

  @parameterized.parameters(
      {'num_batches': 1},
      {'num_batches': 4},
  )
  def test_run_inception_graph(self, num_batches):
    """Test `run_inception` graph construction."""
    batch_size = 8
    img = tf.ones([batch_size, 299, 299, 3])

    results = self.run_inception(img, num_batches=num_batches)

    self.assertIsInstance(results, dict)
    self.assertLen(results, 2)

    self.assertIn('logits', results)
    logits = results['logits']
    self.assertIsInstance(logits, tf.Tensor)
    logits.shape.assert_is_compatible_with([batch_size, 1008])

    self.assertIn('pool_3', results)
    pool = results['pool_3']
    self.assertIsInstance(pool, tf.Tensor)
    pool.shape.assert_is_compatible_with([batch_size, 2048])

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())

  def test_run_inception_multicall(self):
    """Test that `run_inception` can be called multiple times."""
    for batch_size in (7, 3, 2):
      img = tf.ones([batch_size, 299, 299, 3])
      self.run_inception(img)


class SampleAndRunInception(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SampleAndRunInception, self).setUp()
    self.export_path = os.path.join(self.get_temp_dir(), 'my-module')
    tf.saved_model.save(FakeInceptionModule(), self.export_path)
    classifier_fn = tfgan.eval.classifier_fn_from_tfhub(self.export_path, None)
    def sample_and_run_inception(*args, **kwargs):
      return tfgan.eval.sample_and_run_inception(
          *args, classifier_fn=classifier_fn, **kwargs)
    self.sample_and_run_inception = sample_and_run_inception

  @parameterized.parameters(
      {'num_batches': 1},
      {'num_batches': 4},
  )
  def test_sample_and_run_inception_graph(self, num_batches):
    """Test `sample_and_run_inception` graph construction."""
    batch_size = 8
    def sample_fn(_):
      return tf.ones([batch_size, 244, 244, 3])
    sample_inputs = [1] * num_batches

    results = self.sample_and_run_inception(sample_fn, sample_inputs)

    self.assertIsInstance(results, dict)
    self.assertLen(results, 2)

    self.assertIn('logits', results)
    logits = results['logits']
    self.assertIsInstance(logits, tf.Tensor)
    logits.shape.assert_is_compatible_with([batch_size * num_batches, 1008])

    self.assertIn('pool_3', results)
    pool = results['pool_3']
    self.assertIsInstance(pool, tf.Tensor)
    pool.shape.assert_is_compatible_with([batch_size * num_batches, 2048])

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())


class InceptionScore(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(InceptionScore, self).setUp()
    self.export_path = os.path.join(self.get_temp_dir(), 'my-module')
    tf.saved_model.save(FakeInceptionModule(), self.export_path)
    classifier_fn = tfgan.eval.classifier_fn_from_tfhub(
        self.export_path, tfgan.eval.INCEPTION_OUTPUT, True)

    def inception_score(*args, **kwargs):
      return tfgan.eval.inception_score(
          *args, classifier_fn=classifier_fn, **kwargs)
    self.inception_score = inception_score

    def inception_score_streaming(*args, **kwargs):
      return tfgan.eval.inception_score_streaming(
          *args, classifier_fn=classifier_fn, **kwargs)
    self.inception_score_streaming = inception_score_streaming

  @parameterized.parameters(
      {'num_batches': 1, 'streaming': True},
      {'num_batches': 1, 'streaming': False},
      {'num_batches': 3, 'streaming': True},
      {'num_batches': 3, 'streaming': False},
  )
  def test_inception_score_graph(self, num_batches, streaming):
    """Test `inception_score` graph construction."""
    if streaming and tf.executing_eagerly():
      # streaming doesn't work in eager execution.
      return
    img = tf.zeros([6, 299, 299, 3])
    if streaming:
      score, update_op = self.inception_score_streaming(
          img, num_batches=num_batches)
      self.assertIsInstance(update_op, tf.Tensor)
      update_op.shape.assert_has_rank(0)
    else:
      score = self.inception_score(img, num_batches=num_batches)

    self.assertIsInstance(score, tf.Tensor)
    score.shape.assert_has_rank(0)

    # Check that none of the model variables are trainable.
    self.assertEmpty(tf.compat.v1.trainable_variables())


class FrechetInceptionDistance(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(FrechetInceptionDistance, self).setUp()
    self.export_path = os.path.join(self.get_temp_dir(), 'my-module')
    tf.saved_model.save(FakeInceptionModule(), self.export_path)
    classifier_fn = tfgan.eval.classifier_fn_from_tfhub(
        self.export_path, tfgan.eval.INCEPTION_FINAL_POOL, True)

    def frechet_inception_distance(*args, **kwargs):
      return tfgan.eval.frechet_inception_distance(
          *args, classifier_fn=classifier_fn, **kwargs)
    self.frechet_inception_distance = frechet_inception_distance

    def fid_streaming(*args, **kwargs):
      return tfgan.eval.frechet_inception_distance_streaming(
          *args, classifier_fn=classifier_fn, **kwargs)
    self.frechet_inception_distance_streaming = fid_streaming

  @parameterized.parameters(
      {'num_batches': 1, 'streaming': True},
      {'num_batches': 1, 'streaming': False},
      {'num_batches': 3, 'streaming': True},
      {'num_batches': 3, 'streaming': False},
  )
  def test_frechet_inception_distance_graph(self, num_batches, streaming):
    """Test `frechet_inception_distance` graph construction."""
    if streaming and tf.executing_eagerly():
      # streaming doesn't work in eager execution.
      return
    img = tf.ones([6, 299, 299, 3])

    if streaming:
      distance, update_op = self.frechet_inception_distance_streaming(
          img, img, num_batches=num_batches)
      self.assertIsInstance(update_op, tf.Tensor)
      update_op.shape.assert_has_rank(0)
    else:
      distance = self.frechet_inception_distance(
          img, img, num_batches=num_batches)

    self.assertIsInstance(distance, tf.Tensor)
    distance.shape.assert_has_rank(0)

    # Check that none of the model variables are trainable.
    self.assertEmpty(tf.compat.v1.trainable_variables())

if __name__ == '__main__':
  tf.test.main()
