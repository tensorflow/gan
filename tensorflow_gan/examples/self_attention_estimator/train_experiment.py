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

# pylint: disable=line-too-long
"""Trains a Self-Attention GAN using Estimators.

This code is based on the original code from https://arxiv.org/abs/1805.08318.

"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import app
from absl import flags
import tensorflow as tf  # tf
from tensorflow_gan.examples.self_attention_estimator import data_provider
from tensorflow_gan.examples.self_attention_estimator import discriminator as dis_module
from tensorflow_gan.examples.self_attention_estimator import estimator_lib as est_lib
from tensorflow_gan.examples.self_attention_estimator import eval_lib
from tensorflow_gan.examples.self_attention_estimator import generator as gen_module


flags.DEFINE_string('model_dir', '/tmp/tfgan_logdir/sagan-estimator',
                    'Optional location to save model. If `None`, use a '
                    'default provided by tf.Estimator.')


# ML Hparams.
flags.DEFINE_integer(
    'train_batch_size', 32,
    'The number of images in each train batch. From go/tpu-pitfalls: "The '
    'batch size of any model should always be at least 64 (8 per TPU core), '
    'since the TPU always pads the tensors to this size. The ideal batch size '
    'when training on the TPU is 1024 (128 per TPU core), since this '
    'eliminates inefficiencies related to memory transfer and padding."')
flags.DEFINE_integer('z_dim', 128,
                     'Dimensions of the generator noise vector.')
flags.DEFINE_integer('gf_dim', 64, 'Dimensionality of gf. [64]')
flags.DEFINE_integer('df_dim', 64, 'Dimensionality of df. [64]')
flags.DEFINE_float('generator_lr', 0.0001, 'The generator learning rate.')
flags.DEFINE_float('discriminator_lr', 0.0004,
                   'The discriminator learning rate.')
flags.DEFINE_float('beta1', 0.0, 'Momentum term of adam. [0.0]')


# ML Infra.
flags.DEFINE_enum(
    'mode', None, ['train', 'continuous_eval', 'train_and_eval'],
    'Mode to run in. `train` just trains the model. `continuous_eval` '
    'continuously looks for new checkpoints and computes eval metrics and '
    'writes sample outputs to disk. `train_and_eval` does both. '
    'If not set, will deduce mode from the TF_CONFIG environment variable.')
flags.DEFINE_integer('max_number_of_steps', 50000,
                     'The maximum number of train steps.')
flags.DEFINE_integer(
    'train_steps_per_eval', 5000,
    'Number of train steps before writing some sample images.')
flags.DEFINE_integer('num_eval_steps', 32, 'The number of evaluation steps.')
flags.DEFINE_integer('eval_batch_size', 32,
                     'The number of images in each eval batch.')
flags.DEFINE_integer('predict_batch_size', 80,
                     'The number of images in each predict batch.')

# Debugging.
flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or CPU.')
flags.DEFINE_bool('eval_on_tpu', False, 'Whether eval is run on TPU.')
flags.DEFINE_integer(
    'continuous_eval_timeout_secs', None,
    'None, or number of seconds to wait for a checkpoint '
    'before stopping.')
flags.DEFINE_bool('fake_data', False, 'Use false data, for testing')
flags.DEFINE_bool('fake_nets', False, 'Use false networks, for testing')

# TPU params.
flags.DEFINE_integer(
    'tpu_iterations_per_loop', 1000,
    'Steps per interior TPU loop. Should be less than '
    '--train_steps_per_eval.')
flags.DEFINE_bool(
    'use_tpu_estimator', False,
    'Whether to use TPUGANEstimator or GANEstimator. This is useful if, for '
    'instance, we want to run the eval job on GPU.')

FLAGS = flags.FLAGS

HParams = collections.namedtuple('HParams', [
    'train_batch_size',
    'eval_batch_size',
    'predict_batch_size',
    'use_tpu',
    'eval_on_tpu',
    'generator_lr',
    'discriminator_lr',
    'beta1',
    'gf_dim',
    'df_dim',
    'num_classes',
    'shuffle_buffer_size',
    'z_dim',
])


def _verify_dataset_shape(ds, z_dim):
  noise_shape = tf.TensorShape([None, z_dim])
  img_shape = tf.TensorShape([None, 128, 128, 3])
  lbl_shape = tf.TensorShape([None])

  ds.output_shapes[0].assert_is_compatible_with(noise_shape)
  ds.output_shapes[1]['images'].assert_is_compatible_with(img_shape)
  ds.output_shapes[1]['labels'].assert_is_compatible_with(lbl_shape)


def train_eval_input_fn(mode, params):
  """Mode-aware input function."""
  is_train = mode == tf.estimator.ModeKeys.TRAIN
  split = 'train' if is_train else 'validation'

  if FLAGS.use_tpu_estimator:
    bs = params['batch_size']
  else:
    bs = {
        tf.estimator.ModeKeys.TRAIN: FLAGS.train_batch_size,
        tf.estimator.ModeKeys.EVAL: FLAGS.eval_batch_size,
    }[mode]

  if FLAGS.fake_data:
    fake_noise = tf.zeros([bs, params['z_dim']])
    fake_imgs = tf.zeros([bs, 128, 128, 3])
    fake_lbls = tf.zeros([bs], dtype=tf.int32)
    ds = tf.data.Dataset.from_tensors(
        (fake_noise, {'images': fake_imgs, 'labels': fake_lbls}))
    _verify_dataset_shape(ds, params['z_dim'])
    return ds

  images_ds = data_provider.provide_dataset(
      bs,
      shuffle_buffer_size=params['shuffle_buffer_size'],
      split=split)
  images_ds = images_ds.map(
      lambda img, lbl: {'images': img, 'labels': lbl})  # map to dict.

  num_towers = 1
  def _make_noise(_):
    noise = gen_module.make_z_normal(num_towers, bs, params['z_dim'])
    return noise[0]  # one tower
  noise_ds = tf.data.Dataset.from_tensors(0).repeat().map(_make_noise)

  ds = tf.data.Dataset.zip((noise_ds, images_ds))
  _verify_dataset_shape(ds, params['z_dim'])
  return ds


def make_estimator(hparams, save_checkpoints_steps=None):
  """Creates a TPU Estimator."""
  generator = _get_generator(hparams, FLAGS.use_tpu_estimator)
  discriminator = _get_discriminator(hparams, FLAGS.use_tpu_estimator)

  if FLAGS.use_tpu_estimator:
    config = est_lib.get_tpu_run_config_from_flags(
        save_checkpoints_steps, FLAGS)
    return est_lib.get_tpu_estimator(generator, discriminator, hparams, config)
  else:
    config = est_lib.get_run_config_from_flags(save_checkpoints_steps, FLAGS)
    return est_lib.get_gpu_estimator(generator, discriminator, hparams, config)


def run_train(hparams):
  """What to run if `FLAGS.mode=='train'`.

  This function runs the `train` method of TPUEstimator, then writes some
  samples to disk.

  Args:
    hparams: A hyperparameter object.
  """
  estimator = make_estimator(
      hparams, save_checkpoints_steps=FLAGS.train_steps_per_eval)
  tf.compat.v1.logging.info(
      'Training until %i steps...' % FLAGS.max_number_of_steps)
  estimator.train(train_eval_input_fn, max_steps=FLAGS.max_number_of_steps)
  tf.compat.v1.logging.info(
      'Finished training %i steps.' % FLAGS.max_number_of_steps)


def run_continuous_eval(hparams):
  """What to run in continuous eval mode."""
  tf.compat.v1.logging.info('Continuous evaluation.')
  estimator = make_estimator(hparams)
  for ckpt_str in tf.contrib.training.checkpoints_iterator(
      FLAGS.model_dir, timeout=FLAGS.continuous_eval_timeout_secs):
    tf.compat.v1.logging.info('Evaluating checkpoint: %s' % ckpt_str)
    estimator.evaluate(
        train_eval_input_fn, steps=FLAGS.num_eval_steps, name='eval_continuous')
    tf.compat.v1.logging.info('Finished evaluating checkpoint: %s' % ckpt_str)


def run_train_and_eval(hparams):
  """Configure and run the train and estimator jobs."""
  estimator = make_estimator(hparams, FLAGS.train_steps_per_eval)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_eval_input_fn, max_steps=FLAGS.max_number_of_steps)
  tf.compat.v1.logging.info(
      'Training until %i steps...' % FLAGS.max_number_of_steps)

  eval_spec = tf.estimator.EvalSpec(
      name='eval',
      input_fn=train_eval_input_fn,
      steps=FLAGS.num_eval_steps,
      start_delay_secs=60,  # Start evaluating after this many seconds.
      throttle_secs=120,  # Wait this long between evals (seconds).
      )
  tf.compat.v1.logging.info(
      'Num eval steps: %i' % FLAGS.num_eval_steps)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def _get_generator(hparams, use_tpu_estimator):
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

    if FLAGS.fake_nets:
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
        gen_imgs, gen_sparse_class, 'generator', use_tpu_estimator)
    eval_lib.log_and_summarize_variables(generator_vars, 'gvars',
                                         use_tpu_estimator)
    gen_imgs.shape.assert_is_compatible_with([None, 128, 128, 3])

    if mode == tf.estimator.ModeKeys.PREDICT:
      return gen_imgs
    else:
      return {'images': gen_imgs, 'labels': gen_sparse_class}
  return generator


def _get_discriminator(hparams, use_tpu_estimator):
  """Return a TF-GAN compatible discriminator."""
  def discriminator(images_and_lbls, unused_conditioning, mode):
    """TF-GAN compatible discriminator."""
    del unused_conditioning, mode
    images, labels = images_and_lbls['images'], images_and_lbls['labels']
    if FLAGS.fake_nets:
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
        eval_lib.log_and_summarize_variables(discriminator_vars, 'dvars',
                                             use_tpu_estimator)
    logits.shape.assert_is_compatible_with([None, None])
    return logits

  return discriminator


def main(_):
  hparams = HParams(
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size,
      use_tpu=FLAGS.use_tpu,
      eval_on_tpu=FLAGS.eval_on_tpu,
      generator_lr=FLAGS.generator_lr,
      discriminator_lr=FLAGS.discriminator_lr,
      beta1=FLAGS.beta1,
      gf_dim=FLAGS.gf_dim,
      df_dim=FLAGS.df_dim,
      num_classes=1000,
      shuffle_buffer_size=10000,
      z_dim=FLAGS.z_dim,
  )
  if FLAGS.mode == 'train':
    run_train(hparams)
  elif FLAGS.mode == 'continuous_eval':
    run_continuous_eval(hparams)
  elif FLAGS.mode == 'train_and_eval' or FLAGS.mode is None:
    run_train_and_eval(hparams)
  else:
    raise ValueError('Mode not recognized: ', FLAGS.mode)


if __name__ == '__main__':
  app.run(main)
