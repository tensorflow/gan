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

"""Tests for TF-GAN's estimator.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

from absl.testing import parameterized
import numpy as np
import six

import tensorflow as tf
import tensorflow_gan as tfgan

# Private functions to test.
from tensorflow_gan.python.estimator.gan_estimator import extract_gan_loss_args_from_params
from tensorflow_gan.python.estimator.gan_estimator import get_eval_estimator_spec
from tensorflow_gan.python.estimator.gan_estimator import get_gan_model
from tensorflow_gan.python.estimator.gan_estimator import get_predict_estimator_spec
from tensorflow_gan.python.estimator.gan_estimator import get_train_estimator_spec
from tensorflow_gan.python.estimator.gan_estimator import Optimizers


def get_sync_optimizer():
  return tf.compat.v1.train.SyncReplicasOptimizer(
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0),
      replicas_to_aggregate=1)


def get_sync_optimizer_hook_type():
  dummy_opt = get_sync_optimizer()
  dummy_hook = dummy_opt.make_session_run_hook(is_chief=True)
  return type(dummy_hook)


def generator_fn(noise_dict, mode):
  del mode
  noise = noise_dict['x']
  return tf.compat.v1.layers.dense(
      noise, tf.compat.dimension_value(noise.shape[1]))


def discriminator_fn(data, unused_conditioning, mode):
  del unused_conditioning, mode
  return tf.compat.v1.layers.dense(data, 1)


class GetGANModelTest(tf.test.TestCase, parameterized.TestCase):
  """Tests that `GetGANModel` produces the correct model."""

  @parameterized.named_parameters(('train', tf.estimator.ModeKeys.TRAIN),
                                  ('eval', tf.estimator.ModeKeys.EVAL),
                                  ('predict', tf.estimator.ModeKeys.PREDICT))
  def test_get_gan_model(self, mode):
    with tf.Graph().as_default():
      generator_inputs = {'x': tf.ones([3, 4])}
      is_predict = mode == tf.estimator.ModeKeys.PREDICT
      real_data = tf.zeros([3, 4]) if not is_predict else None
      gan_model = get_gan_model(
          mode,
          generator_fn,
          discriminator_fn,
          real_data,
          generator_inputs,
          add_summaries=False)

    self.assertEqual(generator_inputs, gan_model.generator_inputs)
    self.assertIsNotNone(gan_model.generated_data)
    self.assertLen(gan_model.generator_variables, 2)  # 1 FC layer
    self.assertIsNotNone(gan_model.generator_fn)
    if mode == tf.estimator.ModeKeys.PREDICT:
      self.assertIsNone(gan_model.real_data)
      self.assertIsNone(gan_model.discriminator_real_outputs)
      self.assertIsNone(gan_model.discriminator_gen_outputs)
      self.assertIsNone(gan_model.discriminator_variables)
      self.assertIsNone(gan_model.discriminator_scope)
      self.assertIsNone(gan_model.discriminator_fn)
    else:
      self.assertIsNotNone(gan_model.real_data)
      self.assertIsNotNone(gan_model.discriminator_real_outputs)
      self.assertIsNotNone(gan_model.discriminator_gen_outputs)
      self.assertLen(gan_model.discriminator_variables, 2)  # 1 FC layer
      self.assertIsNotNone(gan_model.discriminator_scope)
      self.assertIsNotNone(gan_model.discriminator_fn)


def get_dummy_gan_model():
  # TODO(joelshor): Find a better way of creating a variable scope.
  with tf.compat.v1.variable_scope('generator') as gen_scope:
    gen_var = tf.compat.v1.get_variable('dummy_var', initializer=0.0)
  with tf.compat.v1.variable_scope('discriminator') as dis_scope:
    dis_var = tf.compat.v1.get_variable('dummy_var', initializer=0.0)
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


def dummy_loss_fn(gan_model, add_summaries=True):
  del add_summaries
  return tf.reduce_sum(input_tensor=gan_model.discriminator_real_outputs -
                       gan_model.discriminator_gen_outputs)


def get_metrics(gan_model):
  return {
      'mse_custom_metric':
          tf.compat.v1.metrics.mean_squared_error(gan_model.real_data,
                                                  gan_model.generated_data)
  }


class GetEstimatorSpecTest(tf.test.TestCase, parameterized.TestCase):
  """Tests that the EstimatorSpec is constructed appropriately."""

  @classmethod
  def setUpClass(cls):
    super(GetEstimatorSpecTest, cls).setUpClass()
    cls._generator_optimizer = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    cls._discriminator_optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        1.0)

  def test_get_train_estimator_spec(self):
    with tf.Graph().as_default():
      gan_model = get_dummy_gan_model()
      gan_loss = tfgan.gan_loss(gan_model, dummy_loss_fn, dummy_loss_fn)
      spec = get_train_estimator_spec(
          gan_model,
          gan_loss,
          Optimizers(self._generator_optimizer, self._discriminator_optimizer),
          get_hooks_fn=None,  # use default.
          is_chief=True)

    self.assertEqual(tf.estimator.ModeKeys.TRAIN, spec.mode)
    self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
    self.assertIsNotNone(spec.train_op)
    self.assertIsNotNone(spec.training_hooks)

  def test_get_eval_estimator_spec(self):
    with tf.Graph().as_default():
      gan_model = get_dummy_gan_model()
      gan_loss = tfgan.gan_loss(gan_model, dummy_loss_fn, dummy_loss_fn)
      spec = get_eval_estimator_spec(
          gan_model,
          gan_loss,
          get_eval_metric_ops_fn=get_metrics)

    self.assertEqual(tf.estimator.ModeKeys.EVAL, spec.mode)
    self.assertEqual(gan_model.generated_data, spec.predictions)
    self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
    self.assertIsNotNone(spec.eval_metric_ops)

  def test_get_predict_estimator_spec(self):
    with tf.Graph().as_default():
      gan_model = get_dummy_gan_model()
      spec = get_predict_estimator_spec(gan_model)

    self.assertEqual(tf.estimator.ModeKeys.PREDICT, spec.mode)
    self.assertEqual(gan_model.generated_data, spec.predictions)



class GANEstimatorIntegrationTest(tf.test.TestCase):

  def setUp(self):
    super(GANEstimatorIntegrationTest, self).setUp()
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    super(GANEstimatorIntegrationTest, self).tearDown()
    if self._model_dir:
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(self,
                          train_input_fn,
                          eval_input_fn,
                          predict_input_fn,
                          prediction_size,
                          lr_decay=False):

    def make_opt():
      gstep = tf.compat.v1.train.get_or_create_global_step()
      lr = tf.compat.v1.train.exponential_decay(1.0, gstep, 10, 0.9)
      return tf.compat.v1.train.GradientDescentOptimizer(lr)

    gopt = make_opt if lr_decay else tf.compat.v1.train.GradientDescentOptimizer(
        1.0)
    dopt = make_opt if lr_decay else tf.compat.v1.train.GradientDescentOptimizer(
        1.0)
    est = tfgan.estimator.GANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=gopt,
        discriminator_optimizer=dopt,
        get_eval_metric_ops_fn=get_metrics,
        model_dir=self._model_dir)

    # Train.
    num_steps = 10
    est.train(train_input_fn, steps=num_steps)

    # Evaluate.
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[tf.compat.v1.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))
    self.assertEqual(scores['discriminator_loss'], scores['loss'])
    self.assertIn('mse_custom_metric', six.iterkeys(scores))

    # Predict.
    predictions = np.array([x for x in est.predict(predict_input_fn)])

    self.assertAllEqual(prediction_size, predictions.shape)

  def test_numpy_input_fn(self):
    """Tests complete flow with numpy_input_fn."""
    input_dim = 4
    batch_size = 5
    data = np.zeros([batch_size, input_dim])
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': data}, y=data, batch_size=batch_size, shuffle=False)
    predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        prediction_size=[batch_size, input_dim])

  def test_numpy_input_fn_lrdecay(self):
    """Tests complete flow with numpy_input_fn."""
    input_dim = 4
    batch_size = 5
    data = np.zeros([batch_size, input_dim])
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': data}, y=data, batch_size=batch_size, shuffle=False)
    predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        prediction_size=[batch_size, input_dim],
        lr_decay=True)

  def test_input_fn_from_parse_example(self):
    """Tests complete flow with input_fn constructed from parse_example."""
    input_dim = 4
    batch_size = 6
    data = np.zeros([batch_size, input_dim])

    serialized_examples = []
    for datum in data:
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'x':
                      tf.train.Feature(
                          float_list=tf.train.FloatList(value=datum)),
                  'y':
                      tf.train.Feature(
                          float_list=tf.train.FloatList(value=datum)),
              }))
      serialized_examples.append(example.SerializeToString())

    feature_spec = {
        'x': tf.io.FixedLenFeature([input_dim], tf.float32),
        'y': tf.io.FixedLenFeature([input_dim], tf.float32),
    }

    def _train_input_fn():
      feature_map = tf.io.parse_example(
          serialized=serialized_examples, features=feature_spec)
      features = {'x': feature_map['x']}
      labels = feature_map['y']
      return features, labels

    def _eval_input_fn():
      feature_map = tf.io.parse_example(
          serialized=tf.compat.v1.train.limit_epochs(
              serialized_examples, num_epochs=1),
          features=feature_spec)
      features = {'x': feature_map['x']}
      labels = feature_map['y']
      return features, labels

    def _predict_input_fn():
      feature_map = tf.io.parse_example(
          serialized=tf.compat.v1.train.limit_epochs(
              serialized_examples, num_epochs=1),
          features=feature_spec)
      features = {'x': feature_map['x']}
      return features, None

    self._test_complete_flow(
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        prediction_size=[batch_size, input_dim])


class GANEstimatorWarmStartTest(tf.test.TestCase):

  def setUp(self):
    super(GANEstimatorWarmStartTest, self).setUp()
    self._model_dir = self.get_temp_dir()
    self.new_variable_name = 'new_var'
    self.new_variable_value = [1, 2, 3]

  def tearDown(self):
    super(GANEstimatorWarmStartTest, self).tearDown()
    tf.compat.v1.summary.FileWriterCache.clear()

  def _test_warm_start(self, warm_start_from=None):
    """Tests whether WarmStartSettings work as intended."""

    def generator_with_new_variable(noise_dict, mode):
      tf.compat.v1.get_variable(
          name=self.new_variable_name,
          initializer=self.new_variable_value,
          trainable=True)
      return generator_fn(noise_dict, mode)

    def train_input_fn():
      data = np.zeros([3, 4])
      return {'x': data}, data

    est = tfgan.estimator.GANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(1.0),
        discriminator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(
            1.0),
        model_dir=self._model_dir)

    est.train(train_input_fn, steps=1)

    est_warm = tfgan.estimator.GANEstimator(
        generator_fn=generator_with_new_variable,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(1.0),
        discriminator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(
            1.0),
        model_dir=None if warm_start_from else self._model_dir,
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


class GANEstimatorParamsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(GANEstimatorParamsTest, self).setUp()
    self._model_dir = self.get_temp_dir()

  def tearDown(self):
    super(GANEstimatorParamsTest, self).tearDown()
    tf.compat.v1.summary.FileWriterCache.clear()

  @parameterized.named_parameters(
      ('mi_penalty', 1.0),
      ('no_mi_penalty', None))
  def test_params_used(self, mi_penalty):
    def train_input_fn(params):
      self.assertIn('batch_size', params)
      data = np.zeros([params['batch_size'], 4])
      return {'x': data}, data

    est = tfgan.estimator.GANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(1.0),
        discriminator_optimizer=tf.compat.v1.train.GradientDescentOptimizer(
            1.0),
        model_dir=self._model_dir,
        params={
            'batch_size': 4,
            'mutual_information_penalty_weight': mi_penalty
        })

    if mi_penalty:
      with self.assertRaises(ValueError):
        est.train(train_input_fn, steps=1)
    else:
      est.train(train_input_fn, steps=1)

  def test_extract_gan_loss_args_from_params(self):
    params = {'tensor_pool_fn': 1, 'gradient_penalty_target': 2, 'other': 3}
    gan_loss_args = extract_gan_loss_args_from_params(params)
    self.assertEqual(gan_loss_args, {'tensor_pool_fn': 1,
                                     'gradient_penalty_target': 2})

  def test_extract_gan_loss_args_from_params_forbidden(self):
    params = {'tensor_pool_fn': 1, 'model': 2}
    with self.assertRaises(ValueError):
      extract_gan_loss_args_from_params(params)


if __name__ == '__main__':
  tf.test.main()
