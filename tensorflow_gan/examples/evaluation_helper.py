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

"""Contains functions for evaluation and summarization of metrics.

Copied from tensorflow/python/training/evaluation.py and
third_party/tensorflow/contrib/training/python/training/evaluation.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import tensorflow as tf

from tensorflow.python.training import basic_session_run_hooks  # pylint:disable=g-direct-tensorflow-import


def get_or_create_eval_step():
  """Gets or creates the eval step `Tensor`.

  Returns:
    A `Tensor` representing a counter for the evaluation step.

  Raises:
    ValueError: If multiple `Tensors` have been added to the
      `tf.GraphKeys.EVAL_STEP` collection.
  """
  graph = tf.compat.v1.get_default_graph()
  eval_steps = graph.get_collection(tf.compat.v1.GraphKeys.EVAL_STEP)
  if len(eval_steps) == 1:
    return eval_steps[0]
  elif len(eval_steps) > 1:
    raise ValueError('Multiple tensors added to tf.GraphKeys.EVAL_STEP')
  else:
    counter = tf.compat.v1.get_variable(
        'eval_step',
        shape=[],
        dtype=tf.int64,
        initializer=tf.compat.v1.zeros_initializer(),
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
            tf.compat.v1.GraphKeys.EVAL_STEP
        ])
    return counter


def get_latest_eval_step_value(update_ops):
  """Gets the eval step `Tensor` value after running `update_ops`.

  Args:
    update_ops: A list of `Tensors` or a dictionary of names to `Tensors`,
        which are run before reading the eval step value.

  Returns:
    A `Tensor` representing the value for the evaluation step.
  """
  if isinstance(update_ops, dict):
    update_ops = list(update_ops.values())

  with tf.control_dependencies(update_ops):
    return tf.identity(get_or_create_eval_step().read_value())


class MultiStepStopAfterNEvalsHook(tf.estimator.SessionRunHook):
  """Run hook used by the evaluation routines to run the `eval_ops` N times."""

  def __init__(self, num_evals, steps_per_run=1):
    """Constructs the run hook.

    Args:
      num_evals: The number of evaluations to run for. if set to None, will
        iterate the dataset until all inputs are exhausted.
      steps_per_run: Number of steps executed per run call.
    """
    self._num_evals = num_evals
    self._evals_completed = None
    self._steps_per_run_initial_value = steps_per_run

  def _set_evals_completed_tensor(self, updated_eval_step):
    self._evals_completed = updated_eval_step

  def begin(self):
    self._steps_per_run_variable = \
        basic_session_run_hooks.get_or_create_steps_per_run_variable()

  def after_create_session(self, session, coord):
    # Update number of steps to run in the first run call
    if  self._num_evals is None:
      steps = self._steps_per_run_initial_value
    else:
      steps = min(self._steps_per_run_initial_value, self._num_evals)
    self._steps_per_run_variable.load(steps, session=session)

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(
        {'evals_completed': self._evals_completed})

  def after_run(self, run_context, run_values):
    evals_completed = run_values.results['evals_completed']
    # Update number of steps to run in the next iteration
    if  self._num_evals is None:
      steps = self._steps_per_run_initial_value
    else:
      steps = min(self._num_evals - evals_completed,
                  self._steps_per_run_initial_value)
    self._steps_per_run_variable.load(steps, session=run_context.session)

    if self._num_evals is None:
      tf.compat.v1.logging.info('Evaluation [%d]', evals_completed)
    else:
      tf.compat.v1.logging.info('Evaluation [%d/%d]', evals_completed,
                                self._num_evals)
    if self._num_evals is not None and evals_completed >= self._num_evals:
      run_context.request_stop()


class StopAfterNEvalsHook(tf.estimator.SessionRunHook):
  """Run hook used by the evaluation routines to run the `eval_ops` N times."""

  def __init__(self, num_evals, log_progress=True):
    """Constructs the run hook.

    Args:
      num_evals: The number of evaluations to run for. if set to None, will
        iterate the dataset until all inputs are exhausted.
      log_progress: Whether to log evaluation progress, defaults to True.
    """
    # The number of evals to run for.
    self._num_evals = num_evals
    self._evals_completed = None
    self._log_progress = log_progress
    # Reduce logging frequency if there are 20 or more evaluations.
    self._log_frequency = (1 if (num_evals is None or num_evals < 20)
                           else math.floor(num_evals / 10.))

  def _set_evals_completed_tensor(self, updated_eval_step):
    self._evals_completed = updated_eval_step

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(
        {'evals_completed': self._evals_completed})

  def after_run(self, run_context, run_values):
    evals_completed = run_values.results['evals_completed']
    if self._log_progress:
      if self._num_evals is None:
        tf.compat.v1.logging.info('Evaluation [%d]', evals_completed)
      else:
        if ((evals_completed % self._log_frequency) == 0 or
            (self._num_evals == evals_completed)):
          tf.compat.v1.logging.info('Evaluation [%d/%d]', evals_completed,
                                    self._num_evals)
    if self._num_evals is not None and evals_completed >= self._num_evals:
      run_context.request_stop()


class SummaryAtEndHook(tf.estimator.SessionRunHook):
  """A run hook that saves a summary with the results of evaluation."""

  def __init__(self,
               log_dir=None,
               summary_writer=None,
               summary_op=None,
               feed_dict=None):
    """Constructs the Summary Hook.

    Args:
      log_dir: The directory where the summary events are saved to.  Used only
        when `summary_writer` is not specified.
      summary_writer: A `tf.summary.FileWriter` to write summary
        events with.
      summary_op: The summary op to run. If left as `None`, then all summaries
        in the tf.GraphKeys.SUMMARIES collection are used.
      feed_dict: An optional feed dictionary to use when evaluating the
        summaries.

    Raises:
      ValueError: If both `log_dir` and `summary_writer` are `None`.
    """
    self._summary_op = summary_op
    self._replace_summary_op = summary_op is None
    self._feed_dict = feed_dict
    self._summary_writer = summary_writer
    self._log_dir = log_dir
    if self._log_dir is None and self._summary_writer is None:
      raise ValueError('One of log_dir or summary_writer should be used.')

  def begin(self):
    if self._replace_summary_op:
      # This can still remain None if there are no summaries.
      self._summary_op = tf.compat.v1.summary.merge_all()
    self._global_step = tf.compat.v1.train.get_or_create_global_step()

  def after_create_session(self, session, coord):
    if self._summary_writer is None and self._log_dir:
      self._summary_writer = tf.compat.v1.summary.FileWriterCache.get(
          self._log_dir)

  def end(self, session):
    if self._summary_op is not None:
      global_step = tf.compat.v1.train.global_step(session, self._global_step)
      summary_str = session.run(self._summary_op, self._feed_dict)
      if self._summary_writer:
        self._summary_writer.add_summary(summary_str, global_step)
    if self._summary_writer:
      self._summary_writer.flush()


def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            seconds_to_sleep=1,
                            timeout=None):
  """Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used or `None` if we're expecting
      a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint path, or None if the timeout was reached.
  """
  tf.compat.v1.logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
  stop_time = time.time() + timeout if timeout is not None else None
  while True:
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None or checkpoint_path == last_checkpoint:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      tf.compat.v1.logging.info('Found new checkpoint at %s', checkpoint_path)
      return checkpoint_path


def checkpoints_iterator(checkpoint_dir,
                         min_interval_secs=0,
                         timeout=None,
                         timeout_fn=None):
  """Continuously yield new checkpoint files as they appear.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:

  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.

  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.

  Yields:
    String paths to latest checkpoint files as they arrive.
  """
  checkpoint_path = None
  while True:
    new_checkpoint_path = wait_for_new_checkpoint(
        checkpoint_dir, checkpoint_path, timeout=timeout)
    if new_checkpoint_path is None:
      if not timeout_fn:
        # timed out
        tf.compat.v1.logging.info('Timed-out waiting for a checkpoint.')
        return
      if timeout_fn():
        # The timeout_fn indicated that we are truly done.
        return
      else:
        # The timeout_fn indicated that more checkpoints may come.
        continue
    start = time.time()
    checkpoint_path = new_checkpoint_path
    yield checkpoint_path
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)


def evaluate_once(checkpoint_path,
                  master='',
                  scaffold=None,
                  eval_ops=None,
                  feed_dict=None,
                  final_ops=None,
                  final_ops_feed_dict=None,
                  hooks=None,
                  config=None):
  """Evaluates the model at the given checkpoint path.

  During a single evaluation, the `eval_ops` is run until the session is
  interrupted or requested to finish. This is typically requested via a
  `StopAfterNEvalsHook` which results in `eval_ops` running the requested number
  of times.

  Optionally, a user can pass in `final_ops`, a single `Tensor`, a list of
  `Tensors` or a dictionary from names to `Tensors`. The `final_ops` is
  evaluated a single time after `eval_ops` has finished running and the fetched
  values of `final_ops` are returned. If `final_ops` is left as `None`, then
  `None` is returned.

  One may also consider using a `SummaryAtEndHook` to record summaries after the
  `eval_ops` have run. If `eval_ops` is `None`, the summaries run immediately
  after the model checkpoint has been restored.

  Note that `evaluate_once` creates a local variable used to track the number of
  evaluations run via `get_or_create_eval_step`.
  Consequently, if a custom local init op is provided via a `scaffold`, the
  caller should ensure that the local init op also initializes the eval step.

  Args:
    checkpoint_path: The path to a checkpoint to use for evaluation.
    master: The BNS address of the TensorFlow master.
    scaffold: An tf.train.Scaffold instance for initializing variables and
      restoring variables. Note that `scaffold.init_fn` is used by the function
      to restore the checkpoint. If you supply a custom init_fn, then it must
      also take care of restoring the model from its checkpoint.
    eval_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`, which is run until the session is requested to stop,
      commonly done by a `StopAfterNEvalsHook`.
    feed_dict: The feed dictionary to use when executing the `eval_ops`.
    final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`.
    final_ops_feed_dict: A feed dictionary to use when evaluating `final_ops`.
    hooks: List of `tf.train.SessionRunHook` callbacks which are run inside the
      evaluation loop.
    config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.

  Returns:
    The fetched values of `final_ops` or `None` if `final_ops` is `None`.
  """
  eval_step = get_or_create_eval_step()

  # Prepare the run hooks.
  hooks = list(hooks or [])

  if eval_ops is not None:
    if any(isinstance(h, MultiStepStopAfterNEvalsHook) for h in hooks):
      steps_per_run_variable = \
          basic_session_run_hooks.get_or_create_steps_per_run_variable()
      update_eval_step = tf.compat.v1.assign_add(
          eval_step,
          tf.cast(steps_per_run_variable, dtype=eval_step.dtype),
          use_locking=True)
    else:
      update_eval_step = tf.compat.v1.assign_add(eval_step, 1, use_locking=True)

    if isinstance(eval_ops, dict):
      eval_ops['update_eval_step'] = update_eval_step
    elif isinstance(eval_ops, (tuple, list)):
      eval_ops = list(eval_ops) + [update_eval_step]
    else:
      eval_ops = [eval_ops, update_eval_step]

    eval_step_value = get_latest_eval_step_value(eval_ops)

    for h in hooks:
      if isinstance(h, (StopAfterNEvalsHook, MultiStepStopAfterNEvalsHook)):
        h._set_evals_completed_tensor(eval_step_value)  # pylint: disable=protected-access

  tf.compat.v1.logging.info(
      'Starting evaluation at ' +
      time.strftime('%Y-%m-%dT%H:%M:%SZ', time.localtime()))

  # Prepare the session creator.
  session_creator = tf.compat.v1.train.ChiefSessionCreator(
      scaffold=scaffold,
      checkpoint_filename_with_path=checkpoint_path,
      master=master,
      config=config)

  final_ops_hook = tf.estimator.FinalOpsHook(final_ops, final_ops_feed_dict)
  hooks.append(final_ops_hook)

  with tf.compat.v1.train.MonitoredSession(
      session_creator=session_creator, hooks=hooks) as session:
    if eval_ops is not None:
      while not session.should_stop():
        session.run(eval_ops, feed_dict)

  tf.compat.v1.logging.info(
      'Finished evaluation at ' +
      time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()))
  return final_ops_hook.final_ops_values


def evaluate_repeatedly(checkpoint_dir,
                        master='',
                        scaffold=None,
                        eval_ops=None,
                        feed_dict=None,
                        final_ops=None,
                        final_ops_feed_dict=None,
                        eval_interval_secs=60,
                        hooks=None,
                        config=None,
                        max_number_of_evaluations=None,
                        timeout=None,
                        timeout_fn=None):
  """Repeatedly searches for a checkpoint in `checkpoint_dir` and evaluates it.

  During a single evaluation, the `eval_ops` is run until the session is
  interrupted or requested to finish. This is typically requested via a
  `StopAfterNEvalsHook` which results in `eval_ops` running the requested number
  of times.

  Optionally, a user can pass in `final_ops`, a single `Tensor`, a list of
  `Tensors` or a dictionary from names to `Tensors`. The `final_ops` is
  evaluated a single time after `eval_ops` has finished running and the fetched
  values of `final_ops` are returned. If `final_ops` is left as `None`, then
  `None` is returned.

  One may also consider using a `SummaryAtEndHook` to record summaries after the
  `eval_ops` have run. If `eval_ops` is `None`, the summaries run immediately
  after the model checkpoint has been restored.

  Note that `evaluate_once` creates a local variable used to track the number of
  evaluations run via `get_or_create_eval_step`.
  Consequently, if a custom local init op is provided via a `scaffold`, the
  caller should ensure that the local init op also initializes the eval step.

  Args:
    checkpoint_dir: The directory where checkpoints are stored.
    master: The address of the TensorFlow master.
    scaffold: An tf.train.Scaffold instance for initializing variables and
      restoring variables. Note that `scaffold.init_fn` is used by the function
      to restore the checkpoint. If you supply a custom init_fn, then it must
      also take care of restoring the model from its checkpoint.
    eval_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names to
      `Tensors`, which is run until the session is requested to stop, commonly
      done by a `StopAfterNEvalsHook`.
    feed_dict: The feed dictionary to use when executing the `eval_ops`.
    final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`.
    final_ops_feed_dict: A feed dictionary to use when evaluating `final_ops`.
    eval_interval_secs: The minimum number of seconds between evaluations.
    hooks: List of `tf.estimator.SessionRunHook` callbacks which are run inside
      the evaluation loop.
    config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    max_number_of_evaluations: The maximum times to run the evaluation. If left
      as `None`, then evaluation runs indefinitely.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.

  Returns:
    The fetched values of `final_ops` or `None` if `final_ops` is `None`.
  """
  eval_step = get_or_create_eval_step()

  # Prepare the run hooks.
  hooks = hooks or []

  if eval_ops is not None:
    update_eval_step = tf.compat.v1.assign_add(eval_step, 1)

    for h in hooks:
      if isinstance(h, StopAfterNEvalsHook):
        h._set_evals_completed_tensor(update_eval_step)  # pylint: disable=protected-access

    if isinstance(eval_ops, dict):
      eval_ops['update_eval_step'] = update_eval_step
    elif isinstance(eval_ops, (tuple, list)):
      eval_ops = list(eval_ops) + [update_eval_step]
    else:
      eval_ops = [eval_ops, update_eval_step]

  final_ops_hook = tf.estimator.FinalOpsHook(final_ops, final_ops_feed_dict)
  hooks.append(final_ops_hook)
  num_evaluations = 0
  for checkpoint_path in checkpoints_iterator(
      checkpoint_dir,
      min_interval_secs=eval_interval_secs,
      timeout=timeout,
      timeout_fn=timeout_fn):

    session_creator = tf.compat.v1.train.ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_filename_with_path=checkpoint_path,
        master=master,
        config=config)

    with tf.compat.v1.train.MonitoredSession(
        session_creator=session_creator, hooks=hooks) as session:
      tf.compat.v1.logging.info(
          'Starting evaluation at ' +
          time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
      if eval_ops is not None:
        while not session.should_stop():
          session.run(eval_ops, feed_dict)

      tf.compat.v1.logging.info(
          'Finished evaluation at ' +
          time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
    num_evaluations += 1

    if (max_number_of_evaluations is not None and
        num_evaluations >= max_number_of_evaluations):
      return final_ops_hook.final_ops_values

  return final_ops_hook.final_ops_values
