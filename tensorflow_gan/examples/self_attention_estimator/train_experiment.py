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

"""Trains a Self-Attention GAN using Estimators."""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

import tensorflow as tf  # tf
from tensorflow_gan.examples import evaluation_helper as evaluation
from tensorflow_gan.examples.self_attention_estimator import data_provider
from tensorflow_gan.examples.self_attention_estimator import discriminator as dis_module
from tensorflow_gan.examples.self_attention_estimator import estimator_lib as est_lib
from tensorflow_gan.examples.self_attention_estimator import eval_lib
from tensorflow_gan.examples.self_attention_estimator import generator as gen_module


HParams = collections.namedtuple(
    'HParams',
    [
        'train_batch_size',
        'eval_batch_size',
        'predict_batch_size',
        'generator_lr',
        'discriminator_lr',
        'beta1',
        'gf_dim',
        'df_dim',
        'num_classes',
        'shuffle_buffer_size',
        'z_dim',
        'model_dir',
        'max_number_of_steps',
        'train_steps_per_eval',
        'num_eval_steps',
        'debug_params',
        'tpu_params',
    ])
DebugParams = collections.namedtuple(
    'DebugParams',
    [
        'use_tpu',
        'eval_on_tpu',
        'fake_nets',
        'fake_data',
        'continuous_eval_timeout_secs',
    ])
TPUParams = collections.namedtuple(
    'TPUParams',
    [
        'use_tpu_estimator',
        'tpu_location',
        'gcp_project',
        'tpu_zone',
        'tpu_iterations_per_loop',
    ])


def _verify_dataset_shape(ds, z_dim):
  noise_shape = tf.TensorShape([None, z_dim])
  img_shape = tf.TensorShape([None, 128, 128, 3])
  lbl_shape = tf.TensorShape([None])

  ds_shape = tf.compat.v1.data.get_output_shapes(ds)
  ds_shape[0].assert_is_compatible_with(noise_shape)
  ds_shape[1]['images'].assert_is_compatible_with(img_shape)
  ds_shape[1]['labels'].assert_is_compatible_with(lbl_shape)


def train_eval_input_fn(mode, params):
  """Mode-aware input function."""
  is_train = mode == tf.estimator.ModeKeys.TRAIN
  split = 'train' if is_train else 'validation'

  if params['tpu_params'].use_tpu_estimator:
    bs = params['batch_size']
  else:
    bs = {
        tf.estimator.ModeKeys.TRAIN: params['train_batch_size'],
        tf.estimator.ModeKeys.EVAL: params['eval_batch_size'],
        tf.estimator.ModeKeys.PREDICT: params['predict_batch_size'],
    }[mode]

  if params['debug_params'].fake_data:
    fake_noise = tf.zeros([bs, params['z_dim']])
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.data.Dataset.from_tensors(fake_noise).repeat()
    fake_imgs = tf.zeros([bs, 128, 128, 3])
    fake_lbls = tf.zeros([bs], dtype=tf.int32)
    ds = tf.data.Dataset.from_tensors(
        (fake_noise, {'images': fake_imgs, 'labels': fake_lbls}))
    ds = ds.repeat()
    _verify_dataset_shape(ds, params['z_dim'])
    return ds

  num_towers = 1

  def _make_noise(_):
    noise = gen_module.make_z_normal(num_towers, bs, params['z_dim'])
    return noise[0]  # one tower

  noise_ds = tf.data.Dataset.from_tensors(0).repeat().map(_make_noise)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return noise_ds

  images_ds = data_provider.provide_dataset(
      bs,
      shuffle_buffer_size=params['shuffle_buffer_size'],
      split=split)
  images_ds = images_ds.map(
      lambda img, lbl: {'images': img, 'labels': lbl})  # map to dict.

  ds = tf.data.Dataset.zip((noise_ds, images_ds))
  _verify_dataset_shape(ds, params['z_dim'])
  return ds


def make_estimator(hparams):
  """Creates a TPU Estimator."""
  generator = _get_generator(hparams)
  discriminator = _get_discriminator(hparams)

  if hparams.tpu_params.use_tpu_estimator:
    config = est_lib.get_tpu_run_config_from_hparams(hparams)
    return est_lib.get_tpu_estimator(generator, discriminator, hparams, config)
  else:
    config = est_lib.get_run_config_from_hparams(hparams)
    return est_lib.get_gpu_estimator(generator, discriminator, hparams, config)


def run_train(hparams):
  """What to run if `FLAGS.mode=='train'`.

  This function runs the `train` method of TPUEstimator, then writes some
  samples to disk.

  Args:
    hparams: A hyperparameter object.
  """
  estimator = make_estimator(hparams)
  tf.compat.v1.logging.info('Training until %i steps...' %
                            hparams.max_number_of_steps)
  estimator.train(train_eval_input_fn, max_steps=hparams.max_number_of_steps)
  tf.compat.v1.logging.info('Finished training %i steps.' %
                            hparams.max_number_of_steps)


def run_continuous_eval(hparams):
  """What to run in continuous eval mode."""
  tf.compat.v1.logging.info('Continuous evaluation.')
  estimator = make_estimator(hparams)
  timeout = hparams.debug_params.continuous_eval_timeout_secs
  for ckpt_str in evaluation.checkpoints_iterator(
      hparams.model_dir, timeout=timeout):
    tf.compat.v1.logging.info('Evaluating checkpoint: %s' % ckpt_str)
    estimator.evaluate(
        train_eval_input_fn,
        steps=hparams.num_eval_steps,
        name='eval_continuous')
    tf.compat.v1.logging.info('Finished evaluating checkpoint: %s' % ckpt_str)


# TODO(joelshor): Try to get this to work with
# `tf.estimator.train_and_evaluate`.
def run_train_and_eval(hparams):
  """Configure and run the train and estimator jobs."""
  estimator = make_estimator(hparams)

  # Recover from a previous step, if we've trained at all.
  try:
    cur_step = estimator.get_variable_value('global_step')
  except ValueError:
    cur_step = 0

  max_step = hparams.max_number_of_steps
  steps_per_eval = hparams.train_steps_per_eval

  start_time = time.time()
  while cur_step < max_step:
    if hparams.tpu_params.use_tpu_estimator:
      tf.compat.v1.logging.info('About to write sample images at step: %i' %
                                cur_step)
      eval_lib.predict_and_write_images(estimator, train_eval_input_fn,
                                        hparams.model_dir, 'step_%i' % cur_step)

    # Train for a fixed number of steps.
    start_step = cur_step
    step_to_stop_at = min(cur_step + steps_per_eval, max_step)
    tf.compat.v1.logging.info('About to train to step: %i' % step_to_stop_at)
    start = time.time()
    estimator.train(train_eval_input_fn, max_steps=step_to_stop_at)
    end = time.time()
    cur_step = step_to_stop_at

    # Print some performance statistics.
    steps_taken = step_to_stop_at - start_step
    time_taken = end - start
    _log_performance_statistics(cur_step, steps_taken, time_taken, start_time)

    # Run evaluation.
    tf.compat.v1.logging.info('Evaluating at step: %i' % cur_step)
    estimator.evaluate(
        train_eval_input_fn, steps=hparams.num_eval_steps, name='eval')
    tf.compat.v1.logging.info('Finished evaluating step: %i' % cur_step)


def _log_performance_statistics(cur_step, steps_taken, time_taken, start_time):
  steps_per_sec = steps_taken / time_taken
  min_since_start = (time.time() - start_time) / 60.0
  tf.compat.v1.logging.info(
      'Current step: %i, %.4f steps / sec, time since start: %.1f min' % (
          cur_step, steps_per_sec, min_since_start))


def _get_generator(hparams):
  """Returns a TF-GAN compatible generator function."""
  def generator(noise, mode):
    """TF-GAN compatible generator function."""
    batch_size = tf.shape(input=noise)[0]
    is_train = (mode == tf.estimator.ModeKeys.TRAIN)

    # Some label trickery.
    gen_class_logits = tf.zeros((batch_size, hparams.num_classes))
    gen_class_ints = tf.random.categorical(
        logits=gen_class_logits, num_samples=1)
    gen_sparse_class = tf.squeeze(gen_class_ints, -1)
    gen_sparse_class.shape.assert_is_compatible_with([None])

    if hparams.debug_params.fake_nets:
      gen_imgs = tf.zeros([batch_size, 128, 128, 3
                          ]) * tf.compat.v1.get_variable(
                              'dummy_g', initializer=2.0)
      generator_vars = ()
    else:
      gen_imgs, generator_vars = gen_module.generator(
          noise,
          gen_sparse_class,
          hparams.gf_dim,
          hparams.num_classes,
          training=is_train)
    # Print debug statistics and log the generated variables.
    gen_imgs, gen_sparse_class = eval_lib.print_debug_statistics(
        gen_imgs, gen_sparse_class, 'generator',
        hparams.tpu_params.use_tpu_estimator)
    eval_lib.log_and_summarize_variables(generator_vars, 'gvars',
                                         hparams.tpu_params.use_tpu_estimator)
    gen_imgs.shape.assert_is_compatible_with([None, 128, 128, 3])

    if mode == tf.estimator.ModeKeys.PREDICT:
      return gen_imgs
    else:
      return {'images': gen_imgs, 'labels': gen_sparse_class}
  return generator


def _get_discriminator(hparams):
  """Return a TF-GAN compatible discriminator."""
  def discriminator(images_and_lbls, unused_conditioning, mode):
    """TF-GAN compatible discriminator."""
    del unused_conditioning, mode
    images, labels = images_and_lbls['images'], images_and_lbls['labels']
    if hparams.debug_params.fake_nets:
      # Need discriminator variables and to depend on the generator.
      logits = tf.zeros(
          [tf.shape(input=images)[0], 20]) * tf.compat.v1.get_variable(
              'dummy_d', initializer=2.0) * tf.reduce_mean(input_tensor=images)
      discriminator_vars = ()
    else:
      num_trainable_variables = len(tf.compat.v1.trainable_variables())
      logits, discriminator_vars = dis_module.discriminator(
          images, labels, hparams.df_dim, hparams.num_classes)
      if num_trainable_variables != len(tf.compat.v1.trainable_variables()):
        # Log the generated variables only in the first time the function is
        # called and new variables are generated (it is called twice: once for
        # the generated data and once for the real data).
        eval_lib.log_and_summarize_variables(
            discriminator_vars, 'dvars', hparams.tpu_params.use_tpu_estimator)
    logits.shape.assert_is_compatible_with([None, None])
    return logits

  return discriminator
