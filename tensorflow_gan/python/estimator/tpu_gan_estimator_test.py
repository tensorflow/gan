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
# ============================================================================
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for TF-GAN's TPU Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

# Dependency imports
from absl import flags
from absl.testing import parameterized
import numpy as np
import six

import tensorflow as tf
import tensorflow_gan as tfgan

# Private functions to test.
from tensorflow_gan.python.estimator.tpu_gan_estimator import get_estimator_spec


flags.DEFINE_bool('use_tpu', False, 'Whether to run test on TPU or not.')


def generator_fn(noise, mode):
  del mode
  # TODO(joelshor): Use `tf.compat.dimension_value` when I figure out how to
  # use it in open source.
  return tf.contrib.layers.fully_connected(noise, noise.shape[1].value)


def discriminator_fn(data, unused_conditioning, mode):
  del unused_conditioning, mode
  return tf.contrib.layers.fully_connected(data, 1)


def get_dummy_gan_model():
  # TODO(joelshor): Find a better way of creating a variable scope.
  with tf.variable_scope('generator') as gen_scope:
    gen_var = tf.get_variable('dummy_var', initializer=0.0)
  with tf.variable_scope('discriminator') as dis_scope:
    dis_var = tf.get_variable('dummy_var', initializer=0.0)
  return tfgan.GANModel(
      generator_inputs=None,
      generated_data=tf.ones([3, 4]),
      generator_variables=[gen_var],
      generator_scope=gen_scope,
      generator_fn=None,
      real_data=tf.zeros([3, 4]),
      discriminator_real_outputs=tf.ones([1, 2, 3]) * dis_var,
      discriminator_gen_outputs=tf.ones([1, 2, 3]) * gen_var * dis_var,
      discriminator_variables=[dis_var],
      discriminator_scope=dis_scope,
      discriminator_fn=None)


def get_metrics(generator_inputs, generated_data, real_data,
                discriminator_real_outputs, discriminator_gen_outputs):
  del generator_inputs, discriminator_real_outputs, discriminator_gen_outputs
  return {
      'mse_custom_metric':
          tf.metrics.mean_squared_error(real_data, generated_data)
  }


class GetTPUEstimatorSpecTest(tf.test.TestCase, parameterized.TestCase):
  """Tests that the EstimatorSpec is constructed appropriately."""

  @classmethod
  def setUpClass(cls):
    super(GetTPUEstimatorSpecTest, cls).setUpClass()
    cls._generator_optimizer = tf.contrib.tpu.CrossShardOptimizer(
        tf.train.GradientDescentOptimizer(1.0))
    cls._discriminator_optimizer = tf.contrib.tpu.CrossShardOptimizer(
        tf.train.GradientDescentOptimizer(1.0))

  @parameterized.named_parameters(
      ('joint_train', tf.estimator.ModeKeys.TRAIN, True),
      ('train_sequential', tf.estimator.ModeKeys.TRAIN, False),
      ('eval', tf.estimator.ModeKeys.EVAL, None),
      ('predict', tf.estimator.ModeKeys.PREDICT, None))
  def test_get_estimator_spec(self, mode, joint_train):
    with tf.Graph().as_default():
      self._gan_model = get_dummy_gan_model()
      spec = get_estimator_spec(
          mode,
          self._gan_model,
          generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
          discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
          get_eval_metric_ops_fn=get_metrics,
          generator_optimizer=self._generator_optimizer,
          discriminator_optimizer=self._discriminator_optimizer,
          joint_train=joint_train,
          is_on_tpu=flags.FLAGS.use_tpu,
          gan_train_steps=tfgan.GANTrainSteps(1, 1))

    self.assertIsInstance(spec, tf.contrib.tpu.TPUEstimatorSpec)
    self.assertEqual(mode, spec.mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
      self.assertEqual({'generated_data': self._gan_model.generated_data},
                       spec.predictions)
    elif mode == tf.estimator.ModeKeys.TRAIN:
      self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
      self.assertIsNotNone(spec.train_op)
      self.assertIsNotNone(spec.training_hooks)
    elif mode == tf.estimator.ModeKeys.EVAL:
      self.assertEqual(self._gan_model.generated_data, spec.predictions)
      self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
      self.assertIsNotNone(spec.eval_metrics)


class TPUGANEstimatorIntegrationTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TPUGANEstimatorIntegrationTest, self).setUp()
    self._model_dir = tempfile.mkdtemp()
    self._config = tf.contrib.tpu.RunConfig(model_dir=self._model_dir)

  def tearDown(self):
    super(TPUGANEstimatorIntegrationTest, self).tearDown()
    if self._model_dir:
      tf.summary.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, prediction_size,
      lr_decay=False, joint_train=True):
    def make_opt():
      gstep = tf.train.get_or_create_global_step()
      lr = tf.train.exponential_decay(1.0, gstep, 10, 0.9)
      return tf.train.GradientDescentOptimizer(lr)

    gopt = make_opt if lr_decay else tf.train.GradientDescentOptimizer(1.0)
    dopt = make_opt if lr_decay else tf.train.GradientDescentOptimizer(1.0)
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
    self.assertEqual(num_steps_train + num_steps_eval,
                     scores[tf.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))
    self.assertEqual(scores['discriminator_loss'] + scores['generator_loss'],
                     scores['loss'])
    self.assertIn('mse_custom_metric', six.iterkeys(scores))

    # Predict.
    predictions = np.array([x['generated_data'] for x in
                            est.predict(predict_input_fn)])
    self.assertAllEqual(prediction_size, predictions.shape)

  @parameterized.named_parameters(
      ('joint_train', True, False, False),
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
        x, y = ds.make_one_shot_iterator().get_next()
        return x, y
    def eval_input_fn(params):
      data = np.zeros([input_dim], dtype=np.float32)
      ds = (
          tf.data.Dataset.from_tensors((data, data)).repeat().batch(
              params['batch_size'], drop_remainder=True))
      if return_ds:
        return ds
      else:
        x, y = ds.make_one_shot_iterator().get_next()
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


class TPUGANEstimatorWarmStartTest(tf.test.TestCase):

  def setUp(self):
    self._model_dir = self.get_temp_dir()
    self._config = tf.contrib.tpu.RunConfig(model_dir=self._model_dir)
    self.new_variable_name = 'new_var'
    self.new_variable_value = [1.0, 2.0, 3.0]

  def tearDown(self):
    tf.summary.FileWriterCache.clear()

  def _test_warm_start(self, warm_start_from=None):
    """Tests whether WarmStartSettings work as intended."""
    def generator_with_new_variable(noise_dict, mode):
      tf.get_variable(
          name=self.new_variable_name,
          initializer=self.new_variable_value,
          trainable=True)
      return generator_fn(noise_dict, mode)

    est = tfgan.estimator.TPUGANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.train.GradientDescentOptimizer(1.0),
        discriminator_optimizer=tf.train.GradientDescentOptimizer(1.0),
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
        generator_optimizer=tf.train.GradientDescentOptimizer(1.0),
        discriminator_optimizer=tf.train.GradientDescentOptimizer(1.0),
        config=tf.contrib.tpu.RunConfig(
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
    equal_vals = np.array_equal(est_warm.get_variable_value(full_variable_name),
                                self.new_variable_value)
    self.assertTrue(equal_vals)

if __name__ == '__main__':
  tf.test.main()
