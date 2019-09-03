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

"""Tests for TF-GAN's TPU Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import shutil
import tempfile

from absl import flags
from absl.testing import parameterized
import numpy as np
import six

import tensorflow as tf
import tensorflow_gan as tfgan

# Private functions to test.
from tensorflow_gan.python.estimator.tpu_gan_estimator import get_eval_estimator_spec
from tensorflow_gan.python.estimator.tpu_gan_estimator import get_predict_estimator_spec
from tensorflow_gan.python.estimator.tpu_gan_estimator import get_train_estimator_spec
from tensorflow_gan.python.estimator.tpu_gan_estimator import LossFns
from tensorflow_gan.python.estimator.tpu_gan_estimator import Optimizers

flags.DEFINE_bool('use_tpu', False, 'Whether to run test on TPU or not.')


TpuRunConfig = tf.compat.v1.estimator.tpu.RunConfig
CrossShardOptimizer = tf.compat.v1.tpu.CrossShardOptimizer
TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec


class TestOptimizerWrapper(tf.compat.v1.train.Optimizer):
  """An optimizer wrapper that is designed to share a real optimizer.

  The idea is that multiple instances of this class can share the real optimizer
  and this class will keep track of which steps executed on the real optimizer
  were executed by which instance of the wrapper class. This is useful for
  testing that the order of generator and discriminator steps is as desired.

  This optimizer also has an assertion that two consecutive substeps do not
  generate the same loss. This is meant for the toy case where every substep
  uses the same input data. If the assertion fails it implies that the weights
  for the second step were read before the updates from the first step were
  applied (or the training has converged, which is unlikely in a test scenario).
  """

  def __init__(self, opt, name):
    super(TestOptimizerWrapper, self).__init__(use_locking=False, name=name)
    self._opt = opt
    self._first_call = True
    self._name = name

  def compute_gradients(self, loss, var_list, *args, **kwargs):
    # Ensure that we don't get the same loss twice in a row. If we get this it
    # implies that the previous weight updates have not been applied before the
    # loss was computed.
    if self._first_call:
      self._create_non_slot_variable(
          initial_value=0.0, name='last_loss', colocate_with=var_list[0])

    graph = None if tf.executing_eagerly() else var_list[0].graph
    last_loss = self._get_non_slot_variable('last_loss', graph=graph)

    if self._first_call:
      assert_op = tf.no_op()
    else:
      substep_counter = self._opt._get_non_slot_variable(  # pylint:disable=protected-access
          'substep_counter', graph=graph)
      assert_op = tf.Assert(
          tf.not_equal(loss, last_loss), [
              self._name, 'encountered repeated loss at substep',
              substep_counter, 'current loss:', loss, 'previous loss:',
              last_loss
          ])

    with tf.control_dependencies([assert_op]):
      assign_op = last_loss.assign(loss, use_locking=True)

    self._first_call = False
    with tf.control_dependencies([assign_op]):
      return self._opt.compute_gradients(loss, var_list, *args, **kwargs)

  # Wraps the apply_gradients method of the shared 'real' optimizer, but also
  # updates the internal substep_counter and substep_mask variables to indicate
  # the that the substep was executed on this optimizer. Tests that want to read
  # these variables should access them via Estimator.get_variable_value(), since
  # Estimator.train creates its own tf.Graph, so reading the variables from the
  # optimizer instance would give errors about using a variable in a different
  # Graph than where it was created.
  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    colocate_with = grads_and_vars[0][1]
    # Shared with other wrappers of self._opt.
    self._opt._create_non_slot_variable(  # pylint:disable=protected-access
        initial_value=0,
        name='substep_counter',
        colocate_with=colocate_with)
    # Not shared
    self._create_non_slot_variable(
        initial_value=0,
        name='substep_mask',
        colocate_with=colocate_with)

    update_op = self._opt.apply_gradients(
        grads_and_vars, global_step=global_step)
    graph = None if tf.executing_eagerly() else colocate_with.graph
    with tf.control_dependencies([update_op]):
      return self._track_calls(graph)

  def _track_calls(self, graph):
    substep_counter = self._opt._get_non_slot_variable(  # pylint:disable=protected-access
        'substep_counter', graph=graph)

    substep_mask = self._get_non_slot_variable('substep_mask', graph=graph)

    current_substep_mask = tf.bitwise.left_shift(1, substep_counter)
    updated_substep_mask = tf.bitwise.bitwise_or(current_substep_mask,
                                                 substep_mask)
    assign_op = tf.compat.v1.assign(
        substep_mask, updated_substep_mask, use_locking=True)
    with tf.control_dependencies([assign_op]):
      inc_op = tf.compat.v1.assign_add(substep_counter, 1, use_locking=True)

    return inc_op


def generator_fn(noise, mode):
  del mode
  return tf.compat.v1.layers.dense(
      noise, tf.compat.dimension_value(noise.shape[1]))


def discriminator_fn(data, unused_conditioning, mode):
  del unused_conditioning, mode
  return tf.compat.v1.layers.dense(data, 1)


def get_dummy_gan_model(generated_data=None):
  """Returns a GANModel tuple for testing."""
  if generated_data is None:
    generated_data = tf.ones([3, 4])
  # TODO(joelshor): Find a better way of creating a variable scope.
  with tf.compat.v1.variable_scope(
      'generator', reuse=tf.compat.v1.AUTO_REUSE) as gen_scope:
    gen_var = tf.compat.v1.get_variable('dummy_var', initializer=0.0)
  with tf.compat.v1.variable_scope(
      'discriminator', reuse=tf.compat.v1.AUTO_REUSE) as dis_scope:
    dis_var = tf.compat.v1.get_variable('dummy_var', initializer=0.0)
  return tfgan.GANModel(
      generator_inputs=tf.zeros(shape=()),
      generated_data=generated_data,
      generator_variables=[gen_var],
      generator_scope=gen_scope,
      generator_fn=None,
      real_data=tf.zeros([3, 4]),
      discriminator_real_outputs=tf.ones([1, 2, 3]) * dis_var,
      discriminator_gen_outputs=tf.ones([1, 2, 3]) * gen_var * dis_var,
      discriminator_variables=[dis_var],
      discriminator_scope=dis_scope,
      discriminator_fn=None)


def prepare_arguments_for_metric_fn(generator_inputs, generated_data, real_data,
                                    discriminator_real_outputs,
                                    discriminator_gen_outputs):
  del generator_inputs, discriminator_real_outputs, discriminator_gen_outputs
  return {
      'my_real_data': real_data,
      'my_generated_data': generated_data,
  }


def get_metrics_custom_args(my_real_data, my_generated_data):
  return {
      'mse_custom_metric':
          tf.compat.v1.metrics.mean_squared_error(my_real_data,
                                                  my_generated_data)
  }


def get_metrics(generator_inputs, generated_data, real_data,
                discriminator_real_outputs, discriminator_gen_outputs):
  del generator_inputs, discriminator_real_outputs, discriminator_gen_outputs
  return {
      'mse_custom_metric':
          tf.compat.v1.metrics.mean_squared_error(real_data, generated_data)
  }


class GetTPUEstimatorSpecTest(tf.test.TestCase, parameterized.TestCase):
  """Tests that the EstimatorSpec is constructed appropriately."""

  @classmethod
  def setUpClass(cls):
    super(GetTPUEstimatorSpecTest, cls).setUpClass()
    cls._generator_optimizer = CrossShardOptimizer(
        tf.compat.v1.train.GradientDescentOptimizer(1.0))
    cls._discriminator_optimizer = CrossShardOptimizer(
        tf.compat.v1.train.GradientDescentOptimizer(1.0))
    cls._optimizers = Optimizers(cls._generator_optimizer,
                                 cls._discriminator_optimizer)

    cls._loss_fns = LossFns(tfgan.losses.wasserstein_generator_loss,
                            tfgan.losses.wasserstein_discriminator_loss)

  @parameterized.named_parameters(
      ('joint_train', True),
      ('train_sequential', False),
  )
  def test_get_train_estimator_spec(self, joint_train):
    with tf.Graph().as_default():
      if joint_train:
        gan_model_fns = [get_dummy_gan_model]
      else:
        gan_model_fns = [get_dummy_gan_model, get_dummy_gan_model]
      spec = get_train_estimator_spec(
          gan_model_fns,
          self._loss_fns,
          {},  # gan_loss_kwargs
          self._optimizers,
          joint_train=joint_train,
          is_on_tpu=flags.FLAGS.use_tpu,
          gan_train_steps=tfgan.GANTrainSteps(1, 1),
          add_summaries=not flags.FLAGS.use_tpu)

    self.assertIsInstance(spec, TPUEstimatorSpec)
    self.assertEqual(tf.estimator.ModeKeys.TRAIN, spec.mode)

    self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
    self.assertIsNotNone(spec.train_op)
    self.assertIsNotNone(spec.training_hooks)

  def test_get_eval_estimator_spec(self):
    with tf.Graph().as_default():
      generated_data = tf.ones([3, 4])
      gan_model_fns = [functools.partial(get_dummy_gan_model, generated_data)]
      spec = get_eval_estimator_spec(
          gan_model_fns,
          self._loss_fns,
          gan_loss_kwargs={},
          prepare_arguments_for_eval_metric_fn=None,
          get_eval_metric_ops_fn=get_metrics,
          add_summaries=not flags.FLAGS.use_tpu)

    self.assertIsInstance(spec, TPUEstimatorSpec)
    self.assertEqual(tf.estimator.ModeKeys.EVAL, spec.mode)

    self.assertEqual(generated_data, spec.predictions)
    self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
    self.assertIsNotNone(spec.eval_metrics)

  def test_get_eval_estimator_spec_custom_metric_args(self):
    with tf.Graph().as_default():
      generated_data = tf.ones([3, 4])
      gan_model_fns = [functools.partial(get_dummy_gan_model, generated_data)]
      spec = get_eval_estimator_spec(
          gan_model_fns,
          self._loss_fns,
          gan_loss_kwargs={},
          prepare_arguments_for_eval_metric_fn=prepare_arguments_for_metric_fn,
          get_eval_metric_ops_fn=get_metrics_custom_args,
          add_summaries=not flags.FLAGS.use_tpu)

    self.assertIsInstance(spec, TPUEstimatorSpec)
    self.assertEqual(tf.estimator.ModeKeys.EVAL, spec.mode)

    self.assertEqual(generated_data, spec.predictions)
    self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
    self.assertIsNotNone(spec.eval_metrics)

  def test_get_predict_estimator_spec(self):
    with tf.Graph().as_default():
      generated_data = tf.ones([3, 4])
      gan_model_fns = [functools.partial(get_dummy_gan_model, generated_data)]
      spec = get_predict_estimator_spec(gan_model_fns)

    self.assertIsInstance(spec, TPUEstimatorSpec)
    self.assertEqual(tf.estimator.ModeKeys.PREDICT, spec.mode)
    self.assertEqual({'generated_data': generated_data}, spec.predictions)


class TPUGANEstimatorIntegrationTest(tf.test.TestCase, parameterized.TestCase):
  """Integration tests for TPUGANEstimator."""

  def setUp(self):
    super(TPUGANEstimatorIntegrationTest, self).setUp()
    self._model_dir = tempfile.mkdtemp()
    self._config = TpuRunConfig(model_dir=self._model_dir)

  def tearDown(self):
    super(TPUGANEstimatorIntegrationTest, self).tearDown()
    if self._model_dir:
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(self,
                          train_input_fn,
                          eval_input_fn,
                          predict_input_fn,
                          prediction_size,
                          lr_decay=False,
                          joint_train=True):

    def make_opt():
      gstep = tf.compat.v1.train.get_or_create_global_step()
      lr = tf.compat.v1.train.exponential_decay(1.0, gstep, 10, 0.9)
      return tf.compat.v1.train.GradientDescentOptimizer(lr)

    gopt = make_opt if lr_decay else tf.compat.v1.train.GradientDescentOptimizer(
        1.0)
    dopt = make_opt if lr_decay else tf.compat.v1.train.GradientDescentOptimizer(
        1.0)
    est = tfgan.estimator.TPUGANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=gopt,
        discriminator_optimizer=dopt,
        joint_train=joint_train,
        get_eval_metric_ops_fn=get_metrics,
        train_batch_size=4,
        eval_batch_size=10,
        predict_batch_size=8,
        use_tpu=flags.FLAGS.use_tpu,
        config=self._config)

    # Train.
    num_steps_train = 10
    est.train(train_input_fn, steps=num_steps_train)

    # Evaluate.
    num_steps_eval = 2
    scores = est.evaluate(eval_input_fn, steps=num_steps_eval)
    self.assertIn(tf.compat.v1.GraphKeys.GLOBAL_STEP, six.iterkeys(scores))
    self.assertIn('loss', six.iterkeys(scores))
    self.assertAlmostEqual(
        scores['discriminator_loss'], scores['loss'], places=4)
    self.assertIn('mse_custom_metric', six.iterkeys(scores))

    # Predict.
    predictions = np.array(
        [x['generated_data'] for x in est.predict(predict_input_fn)])
    self.assertAllEqual(prediction_size, predictions.shape)

  @parameterized.named_parameters(('joint_train', True, False, False),
                                  ('train_sequential', False, False, False),
                                  ('lr_decay', False, True, False),
                                  ('train_sequential_ds', False, False, True))
  def test_numpy_input_fn(self, joint_train, lr_decay, return_ds):
    """Tests complete flow with numpy_input_fn."""
    input_dim = 4

    def train_input_fn(params):
      data = np.zeros([input_dim], dtype=np.float32)
      ds = (
          tf.data.Dataset.from_tensors((data, data)).repeat().batch(
              params['batch_size'], drop_remainder=True))
      if return_ds:
        return ds
      else:
        x, y = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
        return x, y

    def eval_input_fn(params):
      data = np.zeros([input_dim], dtype=np.float32)
      ds = (
          tf.data.Dataset.from_tensors((data, data)).repeat().batch(
              params['batch_size'], drop_remainder=True))
      if return_ds:
        return ds
      else:
        x, y = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
        return x, y

    predict_size = 10

    def predict_input_fn(params):
      del params  # unused
      data = np.zeros([input_dim], dtype=np.float32)
      ds = (
          tf.data.Dataset.from_tensors(data).repeat(predict_size).batch(
              1, drop_remainder=True))
      return ds

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        prediction_size=[predict_size, input_dim],
        lr_decay=lr_decay,
        joint_train=joint_train)


class TPUGANEstimatorMultiTrainStepTest(tf.test.TestCase,
                                        parameterized.TestCase):
  """Tests for TPU multistep logic."""

  def setUp(self):
    super(TPUGANEstimatorMultiTrainStepTest, self).setUp()
    self._model_dir = tempfile.mkdtemp()
    self._config = TpuRunConfig(model_dir=self._model_dir)

  def tearDown(self):
    super(TPUGANEstimatorMultiTrainStepTest, self).tearDown()
    if self._model_dir:
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  @parameterized.named_parameters(
      ('1:1 joint', 1, 1, True, 2, [0b10, 0b01], [0b10, 0b01]),
      ('1:1 seq', 1, 1, False, 2, [0b10], [0b01]),
      ('1:3 joint', 1, 3, True, 4, [0b0010, 0b0001], [0b1110, 0b1101]),
      ('1:3 seq', 1, 3, False, 4, [0b1000], [0b0111]),
      ('3:1 joint', 3, 1, True, 4, [0b1110, 0b1101], [0b0010, 0b0001]),
      ('3:1 seq', 3, 1, False, 4, [0b1110], [0b0001]),
      ('1:0 seq', 1, 0, False, 1, [0b1], None),
      ('0:1 seq', 0, 1, False, 1, None, [0b1]))
  def test_train(self, g_steps, d_steps, joint_train, expected_total_substeps,
                 expected_g_substep_mask, expected_d_substep_mask):
    real_opt = tf.compat.v1.train.GradientDescentOptimizer(1e-2)
    gopt = TestOptimizerWrapper(real_opt, name='g_opt')
    dopt = TestOptimizerWrapper(real_opt, name='d_opt')
    est = tfgan.estimator.TPUGANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=gopt,
        discriminator_optimizer=dopt,
        gan_train_steps=tfgan.GANTrainSteps(g_steps, d_steps),
        joint_train=joint_train,
        get_eval_metric_ops_fn=get_metrics,
        train_batch_size=4,
        eval_batch_size=10,
        predict_batch_size=8,
        use_tpu=flags.FLAGS.use_tpu,
        config=self._config)

    def train_input_fn(params):
      data = tf.ones([params['batch_size'], 4], dtype=tf.float32)
      return data, data

    est.train(train_input_fn, steps=1)

    self.assertEqual(1, est.get_variable_value('global_step'))

    substep_counter_name = 'discriminator_train/substep_counter'
    if d_steps == 0:
      substep_counter_name = 'generator_train/substep_counter'
    substep_counter = est.get_variable_value(substep_counter_name)
    self.assertEqual(expected_total_substeps, substep_counter)

    if expected_g_substep_mask is not None:
      g_substep_mask = est.get_variable_value('generator_train/substep_mask')
      self.assertIn(g_substep_mask, expected_g_substep_mask)
    if expected_d_substep_mask is not None:
      d_substep_mask = est.get_variable_value(
          'discriminator_train/substep_mask')
      self.assertIn(d_substep_mask, expected_d_substep_mask)


class TPUGANEstimatorWarmStartTest(tf.test.TestCase):
  """Tests that TPUGANEstimator can be warm-started."""

  def setUp(self):
    super(TPUGANEstimatorWarmStartTest, self).setUp()
    self._model_dir = self.get_temp_dir()
    self._config = TpuRunConfig(model_dir=self._model_dir)
    self.new_variable_name = 'new_var'
    self.new_variable_value = [1.0, 2.0, 3.0]

  def tearDown(self):
    super(TPUGANEstimatorWarmStartTest, self).tearDown()
    tf.compat.v1.summary.FileWriterCache.clear()

  def _test_warm_start(self, warm_start_from=None):
    """Tests whether WarmStartSettings work as intended."""

    def generator_with_new_variable(noise_dict, mode):
      tf.compat.v1.get_variable(
          name=self.new_variable_name,
          initializer=self.new_variable_value,
          trainable=True)
      return generator_fn(noise_dict, mode)

    est = tfgan.estimator.TPUGANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(1.0),
        discriminator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(
            1.0),
        train_batch_size=4,
        use_tpu=flags.FLAGS.use_tpu,
        config=self._config)

    def train_input_fn(params):
      data = tf.zeros([params['batch_size'], 4], dtype=tf.float32)
      return data, data

    est.train(train_input_fn, steps=1)

    est_warm = tfgan.estimator.TPUGANEstimator(
        generator_fn=generator_with_new_variable,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(1.0),
        discriminator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(
            1.0),
        config=TpuRunConfig(
            model_dir=None if warm_start_from else self._model_dir),
        train_batch_size=4,
        use_tpu=flags.FLAGS.use_tpu,
        warm_start_from=warm_start_from)

    est_warm.train(train_input_fn, steps=1)

    return est_warm

  def test_warm_start_error(self):
    """Test if exception when reloading different estimators."""
    with self.assertRaises(tf.errors.NotFoundError):
      self._test_warm_start()

  def test_warm_start_success(self):
    """Test if GANEstimator allows explicit warm start variable assignment."""
    # Regex matches all variable names in ckpt except for new_var.
    var_regex = '^(?!.*%s.*)' % self.new_variable_name
    warmstart = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=self._model_dir, vars_to_warm_start=var_regex)
    est_warm = self._test_warm_start(warm_start_from=warmstart)
    full_variable_name = 'Generator/%s' % self.new_variable_name
    self.assertIn(full_variable_name, est_warm.get_variable_names())
    equal_vals = np.array_equal(
        est_warm.get_variable_value(full_variable_name),
        self.new_variable_value)
    self.assertTrue(equal_vals)
