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

"""Some utilities for self-attention estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from tensorflow_gan.examples.self_attention_estimator import eval_lib
import tensorflow_gan as tfgan  # tf


def get_tpu_run_config_from_hparams(hparams):
  """Create a TPU-suitable RunConfig from HParams."""
  tf.compat.v1.logging.info('tpu_location: %s', hparams.tpu_params.tpu_location)
  tf.compat.v1.logging.info('gcp_project: %s', hparams.tpu_params.gcp_project)
  tf.compat.v1.logging.info('tpu_zone: %s', hparams.tpu_params.tpu_zone)
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=hparams.tpu_params.tpu_location,
      project=hparams.tpu_params.gcp_project,
      zone=hparams.tpu_params.tpu_zone)
  if hparams.debug_params.eval_on_tpu:
    eval_training_input_configuration = tf.compat.v1.estimator.tpu.InputPipelineConfig.SLICED
  else:
    # InputPipelineConfig.SLICED is not supported when running on CPU.
    eval_training_input_configuration = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V1
  return tf.compat.v1.estimator.tpu.RunConfig(
      model_dir=hparams.model_dir,
      cluster=cluster_resolver,
      save_checkpoints_steps=hparams.train_steps_per_eval,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=hparams.tpu_params.tpu_iterations_per_loop,
          eval_training_input_configuration=eval_training_input_configuration))


def get_run_config_from_hparams(hparams):
  mirrored_strategy = tf.distribute.MirroredStrategy()
  return tf.estimator.RunConfig(
      model_dir=hparams.model_dir,
      save_checkpoints_steps=hparams.train_steps_per_eval,
      train_distribute=mirrored_strategy)


def get_tpu_estimator(generator, discriminator, hparams, config):
  return tfgan.estimator.TPUGANEstimator(
      generator_fn=generator,
      discriminator_fn=discriminator,
      generator_loss_fn=tfgan.losses.wasserstein_hinge_generator_loss,
      discriminator_loss_fn=tfgan.losses.wasserstein_hinge_discriminator_loss,
      generator_optimizer=tf.compat.v1.train.AdamOptimizer(
          hparams.generator_lr, hparams.beta1),
      discriminator_optimizer=tf.compat.v1.train.AdamOptimizer(
          hparams.discriminator_lr, hparams.beta1),
      prepare_arguments_for_eval_metric_fn=prepare_metric_arguments,
      get_eval_metric_ops_fn=functools.partial(get_metrics, hparams=hparams),
      eval_on_tpu=hparams.debug_params.eval_on_tpu,
      train_batch_size=hparams.train_batch_size,
      eval_batch_size=hparams.eval_batch_size,
      predict_batch_size=hparams.predict_batch_size,
      use_tpu=hparams.debug_params.use_tpu,
      config=config,
      params=hparams._asdict())


def get_gpu_estimator(generator, discriminator, hparams, config):
  """Returns an Estimator object to be used for training with GPUs."""

  def gpu_get_metric(gan_model):
    """A function compatible with GANEstimator's get_eval_metric_ops_fn arg."""
    metrics_arguments = prepare_metric_arguments(
        gan_model.generator_inputs, gan_model.generated_data,
        gan_model.real_data, gan_model.discriminator_real_outputs,
        gan_model.discriminator_gen_outputs)
    metrics = get_metrics(hparams=hparams, **metrics_arguments)
    # Generate image summaries.
    real_data = gan_model.real_data
    generated_data = gan_model.generated_data
    real_images = (
        real_data['images'] if isinstance(real_data, dict) else real_data)
    gen_images = (
        generated_data['images']
        if isinstance(generated_data, dict) else generated_data)
    metrics.update(_generator_summary_ops(gen_images, real_images))
    return metrics

  return tfgan.estimator.GANEstimator(
      generator_fn=generator,
      discriminator_fn=discriminator,
      generator_loss_fn=tfgan.losses.wasserstein_hinge_generator_loss,
      discriminator_loss_fn=tfgan.losses.wasserstein_hinge_discriminator_loss,
      generator_optimizer=tf.compat.v1.train.AdamOptimizer(
          hparams.generator_lr, hparams.beta1),
      discriminator_optimizer=tf.compat.v1.train.AdamOptimizer(
          hparams.discriminator_lr, hparams.beta1),
      get_eval_metric_ops_fn=gpu_get_metric,
      config=config,
      params=hparams._asdict())


def prepare_metric_arguments(generator_inputs, generated_data, real_data,
                             discriminator_real_outputs,
                             discriminator_gen_outputs):
  """Prepares the arguments needed for get_metrics.

  When training on TPUs, this function should be executed on TPU.

  Args:
    generator_inputs: Inputs to the generator fn.
    generated_data: Output from the generator.
    real_data: A sample of real data.
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data.

  Returns:
    A metric dictionary.
  """
  del generator_inputs, discriminator_real_outputs, discriminator_gen_outputs

  real_images = (real_data['images'] if isinstance(real_data, dict) else
                 real_data)
  gen_images = (generated_data['images'] if isinstance(generated_data, dict)
                else generated_data)
  # Get logits and pools for real and generated images.
  real_logits, real_pools = eval_lib.get_activations(
      lambda: real_images, num_batches=1, get_logits=True)
  fake_logits, fake_pools = eval_lib.get_activations(
      lambda: gen_images, num_batches=1, get_logits=True)

  return {
      'real_logits': real_logits,
      'real_pools': real_pools,
      'fake_logits': fake_logits,
      'fake_pools': fake_pools
  }


def get_metrics(real_logits, real_pools, fake_logits, fake_pools, hparams):
  """Return metrics for SAGAN experiment on TPU, CPU, or GPU.

  When training on TPUs, this function should be executed on the CPU.

  Args:
    real_logits: The real_logits object retured by prepare_metric_arguments.
    real_pools: The real_pools object retured by prepare_metric_arguments.
    fake_logits: The fake_logits object retured by prepare_metric_arguments.
    fake_pools: The fake_pools object retured by prepare_metric_arguments.
    hparams: An hparams object.

  Returns:
    A metric dictionary.
  """
  del hparams
  metric_dict = {
      'eval/real_incscore':
          tfgan.eval.classifier_score_from_logits_streaming(real_logits),
      'eval/incscore':
          tfgan.eval.classifier_score_from_logits_streaming(fake_logits),
      'eval/fid':
          tfgan.eval.frechet_classifier_distance_from_activations_streaming(
              real_pools, fake_pools),
  }
  return metric_dict


def _generator_summary_ops(generated_images, real_images):
  """Creates a dictionary of image summaries."""
  real_img_summ = tf.compat.v1.summary.image('real_images', real_images)
  gen_img_summ = tf.compat.v1.summary.image('gen_images', generated_images)
  real_img_grid = tf.compat.v1.summary.image(
      'real_images_grid',
      tfgan.eval.image_grid(
          real_images[:16],
          grid_shape=(4, 4),
          image_shape=(128, 128),
          num_channels=3))
  gen_img_grid = tf.compat.v1.summary.image(
      'generated_images_grid',
      tfgan.eval.image_grid(
          generated_images[:16],
          grid_shape=(4, 4),
          image_shape=(128, 128),
          num_channels=3))
  return {
      'images/real': (real_img_summ, tf.no_op()),
      'images/gen': (gen_img_summ, tf.no_op()),
      'image_grid/real': (real_img_grid, tf.no_op()),
      'image_grid/gen': (gen_img_grid, tf.no_op()),
  }
