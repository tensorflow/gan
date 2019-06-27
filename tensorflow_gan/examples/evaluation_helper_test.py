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

"""Tests for evaluation_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import time

import numpy as np

import tensorflow as tf

from tensorflow_gan.examples import evaluation_helper as evaluation
from tensorflow_gan.python import contrib_utils as contrib


def _local_variable(val, name):
  return tf.Variable(
      val, name=name, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])


class CheckpointIteratorTest(tf.test.TestCase):

  def testReturnsEmptyIfNoCheckpointsFound(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'no_checkpoints_found')

    num_found = 0
    for _ in evaluation.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 0)

  def testReturnsSingleCheckpointIfOneCheckpointFound(self):
    if tf.executing_eagerly():
      return
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'one_checkpoint_found')
    if not tf.io.gfile.exists(checkpoint_dir):
      tf.io.gfile.makedirs(checkpoint_dir)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    saver = tf.compat.v1.train.Saver()  # Saves the global step.

    with self.cached_session() as session:
      session.run(tf.compat.v1.global_variables_initializer())
      save_path = os.path.join(checkpoint_dir, 'model.ckpt')
      saver.save(session, save_path, global_step=global_step)

    num_found = 0
    for _ in evaluation.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 1)

  def testReturnsSingleCheckpointIfOneShardedCheckpoint(self):
    if tf.executing_eagerly():
      return
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'one_checkpoint_found_sharded')
    if not tf.io.gfile.exists(checkpoint_dir):
      tf.io.gfile.makedirs(checkpoint_dir)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # This will result in 3 different checkpoint shard files.
    with tf.device('/cpu:0'):
      tf.Variable(10, name='v0')
    with tf.device('/cpu:1'):
      tf.Variable(20, name='v1')

    saver = tf.compat.v1.train.Saver(sharded=True)

    with tf.compat.v1.Session(
        target='',
        config=tf.compat.v1.ConfigProto(device_count={'CPU': 2})) as session:

      session.run(tf.compat.v1.global_variables_initializer())
      save_path = os.path.join(checkpoint_dir, 'model.ckpt')
      saver.save(session, save_path, global_step=global_step)

    num_found = 0
    for _ in evaluation.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 1)

  def testTimeoutFn(self):
    timeout_fn_calls = [0]
    def timeout_fn():
      timeout_fn_calls[0] += 1
      return timeout_fn_calls[0] > 3

    results = list(
        evaluation.checkpoints_iterator(
            '/non-existent-dir', timeout=0.1, timeout_fn=timeout_fn))
    self.assertEqual([], results)
    self.assertEqual(4, timeout_fn_calls[0])


def logistic_classifier(inputs):
  return tf.compat.v1.layers.dense(inputs, 1, activation=tf.sigmoid)


class EvaluateOnceTest(tf.test.TestCase):

  def setUp(self):
    super(EvaluateOnceTest, self).setUp()

    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def _train_model(self, checkpoint_dir, num_steps):
    """Trains a simple classification model.

    Note that the data has been configured such that after around 300 steps,
    the model has memorized the dataset (e.g. we can expect %100 accuracy).

    Args:
      checkpoint_dir: The directory where the checkpoint is written to.
      num_steps: The number of steps to train for.
    """
    with tf.Graph().as_default():
      tf.compat.v1.random.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = logistic_classifier(tf_inputs)
      loss = tf.compat.v1.losses.log_loss(tf_predictions, tf_labels)

      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = contrib.create_train_op(loss, optimizer)

      with tf.compat.v1.train.MonitoredTrainingSession(
          hooks=[tf.estimator.StopAtStepHook(num_steps)],
          checkpoint_dir=checkpoint_dir) as sess:
        loss = None
        while not sess.should_stop():
          loss = sess.run(train_op)

      if num_steps >= 300:
        assert loss < .015

  def testEvaluatePerfectModel(self):
    if tf.executing_eagerly():
      # tf.metrics.accuracy is not supported when eager execution is enabled.
      return
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluate_perfect_model_once')

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Run
    inputs = tf.constant(self._inputs, dtype=tf.float32)
    labels = tf.constant(self._labels, dtype=tf.float32)
    logits = logistic_classifier(inputs)
    predictions = tf.round(logits)

    accuracy, update_op = tf.compat.v1.metrics.accuracy(
        predictions=predictions, labels=labels)

    checkpoint_path = evaluation.wait_for_new_checkpoint(checkpoint_dir)

    final_ops_values = evaluation.evaluate_once(
        checkpoint_path=checkpoint_path,
        eval_ops=update_op,
        final_ops={'accuracy': accuracy},
        hooks=[
            evaluation.StopAfterNEvalsHook(1),
        ])
    self.assertTrue(final_ops_values['accuracy'] > .99)

  def testEvalOpAndFinalOp(self):
    if tf.executing_eagerly():
      return
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'eval_ops_and_final_ops')

    # Train a model for a single step to get a checkpoint.
    self._train_model(checkpoint_dir, num_steps=1)
    checkpoint_path = evaluation.wait_for_new_checkpoint(checkpoint_dir)

    # Create the model so we have something to restore.
    inputs = tf.constant(self._inputs, dtype=tf.float32)
    logistic_classifier(inputs)

    num_evals = 5
    final_increment = 9.0

    try:
      my_var = _local_variable(0.0, name='MyVar')
    except TypeError:  # `collections` doesn't exist in TF2.
      return
    eval_ops = tf.compat.v1.assign_add(my_var, 1.0)
    final_ops = tf.identity(my_var) + final_increment

    final_ops_values = evaluation.evaluate_once(
        checkpoint_path=checkpoint_path,
        eval_ops=eval_ops,
        final_ops={'value': final_ops},
        hooks=[
            evaluation.StopAfterNEvalsHook(num_evals),
        ])
    self.assertEqual(final_ops_values['value'], num_evals + final_increment)

  def testOnlyFinalOp(self):
    if tf.executing_eagerly():
      return
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'only_final_ops')

    # Train a model for a single step to get a checkpoint.
    self._train_model(checkpoint_dir, num_steps=1)
    checkpoint_path = evaluation.wait_for_new_checkpoint(checkpoint_dir)

    # Create the model so we have something to restore.
    inputs = tf.constant(self._inputs, dtype=tf.float32)
    logistic_classifier(inputs)

    final_increment = 9.0

    try:
      my_var = _local_variable(0.0, name='MyVar')
    except TypeError:  # `collections` doesn't exist in TF2.
      return
    final_ops = tf.identity(my_var) + final_increment

    final_ops_values = evaluation.evaluate_once(
        checkpoint_path=checkpoint_path, final_ops={'value': final_ops})
    self.assertEqual(final_ops_values['value'], final_increment)


class EvaluateRepeatedlyTest(tf.test.TestCase):

  def setUp(self):
    super(EvaluateRepeatedlyTest, self).setUp()

    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def _train_model(self, checkpoint_dir, num_steps):
    """Trains a simple classification model.

    Note that the data has been configured such that after around 300 steps,
    the model has memorized the dataset (e.g. we can expect %100 accuracy).

    Args:
      checkpoint_dir: The directory where the checkpoint is written to.
      num_steps: The number of steps to train for.
    """
    with tf.Graph().as_default():
      tf.compat.v1.random.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = logistic_classifier(tf_inputs)
      loss = tf.compat.v1.losses.log_loss(tf_predictions, tf_labels)

      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = contrib.create_train_op(loss, optimizer)

      with tf.compat.v1.train.MonitoredTrainingSession(
          hooks=[tf.estimator.StopAtStepHook(num_steps)],
          checkpoint_dir=checkpoint_dir) as sess:
        loss = None
        while not sess.should_stop():
          loss = sess.run(train_op)

  def testEvaluatePerfectModel(self):
    if tf.executing_eagerly():
      return
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluate_perfect_model_repeated')

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Run
    inputs = tf.constant(self._inputs, dtype=tf.float32)
    labels = tf.constant(self._labels, dtype=tf.float32)
    logits = logistic_classifier(inputs)
    predictions = tf.round(logits)

    accuracy, update_op = tf.compat.v1.metrics.accuracy(
        predictions=predictions, labels=labels)

    final_values = evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        eval_ops=update_op,
        final_ops={'accuracy': accuracy},
        hooks=[
            evaluation.StopAfterNEvalsHook(1),
        ],
        max_number_of_evaluations=1)
    self.assertTrue(final_values['accuracy'] > .99)

  def testEvaluationLoopTimeout(self):
    if tf.executing_eagerly():
      # This test uses `tf.placeholder`, which doesn't work in eager executing.
      return
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluation_loop_timeout')
    if not tf.io.gfile.exists(checkpoint_dir):
      tf.io.gfile.makedirs(checkpoint_dir)

    # We need a variable that the saver will try to restore.
    tf.compat.v1.train.get_or_create_global_step()

    # Run with placeholders. If we actually try to evaluate this, we'd fail
    # since we're not using a feed_dict.
    cant_run_op = tf.compat.v1.placeholder(dtype=tf.float32)

    start = time.time()
    final_values = evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        eval_ops=cant_run_op,
        hooks=[evaluation.StopAfterNEvalsHook(10)],
        timeout=6)
    end = time.time()
    self.assertFalse(final_values)

    # Assert that we've waited for the duration of the timeout (minus the sleep
    # time).
    self.assertGreater(end - start, 5.0)

    # Then the timeout kicked in and stops the loop.
    self.assertLess(end - start, 7)

  def testEvaluationLoopTimeoutWithTimeoutFn(self):
    if tf.executing_eagerly():
      # tf.metrics.accuracy is not supported when eager execution is enabled.
      return
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluation_loop_timeout_with_timeout_fn')

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Run
    inputs = tf.constant(self._inputs, dtype=tf.float32)
    labels = tf.constant(self._labels, dtype=tf.float32)
    logits = logistic_classifier(inputs)
    predictions = tf.round(logits)

    accuracy, update_op = tf.compat.v1.metrics.accuracy(
        predictions=predictions, labels=labels)

    timeout_fn_calls = [0]
    def timeout_fn():
      timeout_fn_calls[0] += 1
      return timeout_fn_calls[0] > 3

    final_values = evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        eval_ops=update_op,
        final_ops={'accuracy': accuracy},
        hooks=[
            evaluation.StopAfterNEvalsHook(1),
        ],
        eval_interval_secs=1,
        max_number_of_evaluations=2,
        timeout=0.1,
        timeout_fn=timeout_fn)
    # We should have evaluated once.
    self.assertTrue(final_values['accuracy'] > .99)
    # And called 4 times the timeout fn
    self.assertEqual(4, timeout_fn_calls[0])

  def testEvaluateWithEvalFeedDict(self):
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    # Create a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluate_with_eval_feed_dict')
    self._train_model(checkpoint_dir, num_steps=1)

    # We need a variable that the saver will try to restore.
    tf.compat.v1.train.get_or_create_global_step()

    # Create a variable and an eval op that increments it with a placeholder.
    try:
      my_var = _local_variable(0.0, name='my_var')
    except TypeError:  # `collections` doesn't exist in TF2.
      return
    increment = tf.compat.v1.placeholder(dtype=tf.float32)
    eval_ops = tf.compat.v1.assign_add(my_var, increment)

    increment_value = 3
    num_evals = 5
    expected_value = increment_value * num_evals
    final_values = evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        eval_ops=eval_ops,
        feed_dict={increment: 3},
        final_ops={'my_var': tf.identity(my_var)},
        hooks=[
            evaluation.StopAfterNEvalsHook(num_evals),
        ],
        max_number_of_evaluations=1)
    self.assertEqual(final_values['my_var'], expected_value)

  def _verify_events(self, output_dir, names_to_values):
    """Verifies that the given `names_to_values` are found in the summaries.

    Also checks that a GraphDef was written out to the events file.

    Args:
      output_dir: An existing directory where summaries are found.
      names_to_values: A dictionary of strings to values.
    """
    # Check that the results were saved. The events file may have additional
    # entries, e.g. the event version stamp, so have to parse things a bit.
    output_filepath = glob.glob(os.path.join(output_dir, '*'))
    self.assertEqual(len(output_filepath), 1)

    events = tf.compat.v1.train.summary_iterator(output_filepath[0])
    summaries = []
    graph_def = None
    for event in events:
      if event.summary.value:
        summaries.append(event.summary)
      elif event.graph_def:
        graph_def = event.graph_def
    values = []
    for summary in summaries:
      for value in summary.value:
        values.append(value)
    saved_results = {v.tag: v.simple_value for v in values}
    for name in names_to_values:
      self.assertAlmostEqual(names_to_values[name], saved_results[name], 5)
    self.assertIsNotNone(graph_def)

  def testSummariesAreFlushedToDisk(self):
    if tf.executing_eagerly():
      # Merging tf.summary.* ops is not compatible with eager execution.
      return
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'summaries_are_flushed')
    logdir = os.path.join(self.get_temp_dir(), 'summaries_are_flushed_eval')
    if tf.io.gfile.exists(logdir):
      tf.io.gfile.rmtree(logdir)

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Create the model (which can be restored).
    inputs = tf.constant(self._inputs, dtype=tf.float32)
    logistic_classifier(inputs)

    names_to_values = {'bread': 3.4, 'cheese': 4.5, 'tomato': 2.0}

    for k in names_to_values:
      v = names_to_values[k]
      tf.compat.v1.summary.scalar(k, v)

    evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        hooks=[
            evaluation.SummaryAtEndHook(log_dir=logdir),
        ],
        max_number_of_evaluations=1)

    self._verify_events(logdir, names_to_values)

  def testSummaryAtEndHookWithoutSummaries(self):
    logdir = os.path.join(self.get_temp_dir(),
                          'summary_at_end_hook_without_summaires')
    if tf.io.gfile.exists(logdir):
      tf.io.gfile.rmtree(logdir)

    with tf.Graph().as_default():
      # Purposefully don't add any summaries. The hook will just dump the
      # GraphDef event.
      hook = evaluation.SummaryAtEndHook(log_dir=logdir)
      hook.begin()
      with self.cached_session() as session:
        hook.after_create_session(session, None)
        hook.end(session)
    self._verify_events(logdir, {})


if __name__ == '__main__':
  tf.test.main()
