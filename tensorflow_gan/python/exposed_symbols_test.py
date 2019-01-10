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
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for exposed symbols in tensorflow_gan."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from absl.testing import absltest

import tensorflow_gan as tfgan


UNAVOIDABLE_SYMBOLS = set(['__builtins__', '__doc__', '__file__', '__name__',
                           '__package__', '__path__'])


EXPECTED_MODULE_SYMBOLS = [
    'estimator',
    'eval',
    'features',
    'losses',
]

EXPECTED_NAMEDTUPLES_SYMBOLS = [
    'GANModel',
    'ACGANModel',
    'InfoGANModel',
    'GANLoss',
    'StarGANModel',
    'GANTrainSteps',
    'CycleGANLoss',
    'GANTrainOps',
    'CycleGANModel',
]

EXPECTED_TRAIN_SYMBOLS = [
    'stargan_model',
    'get_joint_train_hooks',
    'stargan_loss',
    'gan_model',
    'cyclegan_model',
    'acgan_model',
    'get_sequential_train_steps',
    'gan_loss',
    'get_sequential_train_hooks',
    'RunTrainOpsHook',
    'cyclegan_loss',
    'infogan_model',
    'gan_train',
    'gan_train_ops',
]


EXPECTED_ESTIMATOR_SYMBOLS = [
    'StarGANEstimator',
    'SummaryType',
    'TPUGANEstimator',
    'GANEstimator',
    'stargan_prediction_input_fn_wrapper',
    'get_latent_gan_estimator'
]


EXPECTED_EVAL_SYMBOLS = [
    'add_cyclegan_image_summaries',
    'kernel_classifier_distance_and_std_from_activations',
    'add_gan_model_summaries',
    'add_regularization_loss_summaries',
    'get_graph_def_from_resource',
    'diagonal_only_frechet_classifier_distance_from_activations',
    'frechet_inception_distance',
    'add_image_comparison_summaries',
    'image_grid',
    'get_graph_def_from_disk',
    'run_inception',
    'sliced_wasserstein_distance',
    'mean_only_frechet_classifier_distance_from_activations',
    'add_gan_model_image_summaries',
    'kernel_inception_distance_and_std',
    'get_graph_def_from_url_tarball',
    'add_stargan_image_summaries',
    'run_image_classifier',
    'kernel_classifier_distance',
    'kernel_inception_distance',
    'frechet_classifier_distance',
    'inception_score',
    'kernel_classifier_distance_and_std',
    'kernel_classifier_distance_from_activations',
    'classifier_score_from_logits',
    'classifier_score',
    'image_reshaper',
    'preprocess_image',
    'frechet_classifier_distance_from_activations',
    'INCEPTION_DEFAULT_IMAGE_SIZE',
]


EXPECTED_FEATURES_SYMBOLS = [
    'VBN',
    'clip_discriminator_weights',
    'clip_variables',
    'condition_tensor',
    'condition_tensor_from_onehot',
    'tensor_pool',
    'compute_spectral_norm',
    'spectral_normalize',
    'spectral_norm_regularizer',
    'spectral_normalization_custom_getter',
]

EXPECTED_LOSSES_SYMBOLS = [
    'minimax_discriminator_loss',
    'combine_adversarial_loss',
    'stargan_discriminator_loss_wrapper',
    'modified_discriminator_loss',
    'stargan_generator_loss_wrapper',
    'mutual_information_penalty',
    'acgan_generator_loss',
    'stargan_gradient_penalty_wrapper',
    'wasserstein_generator_loss',
    'minimax_generator_loss',
    'wasserstein_gradient_penalty',
    'wargs',
    'modified_generator_loss',
    'least_squares_generator_loss',
    'least_squares_discriminator_loss',
    'wasserstein_discriminator_loss',
    'acgan_discriminator_loss',
    'cycle_consistency_loss',
]


EXPECTED_LOSSES_WARGS_SYMBOLS = [
    'acgan_discriminator_loss',
    'wasserstein_gradient_penalty',
    'wasserstein_discriminator_loss',
    'minimax_generator_loss',
    'least_squares_generator_loss',
    'cycle_consistency_loss',
    'acgan_generator_loss',
    'least_squares_discriminator_loss',
    'wasserstein_generator_loss',
    'modified_discriminator_loss',
    'modified_generator_loss',
    'minimax_discriminator_loss',
    'mutual_information_penalty',
    'combine_adversarial_loss',
]


class ExposedSymbolsTest(absltest.TestCase):

  def test_high_level_symbols(self):
    symbols = set([name for name, _ in inspect.getmembers(tfgan)])
    self.assertSetEqual(
        symbols - UNAVOIDABLE_SYMBOLS,
        set(EXPECTED_MODULE_SYMBOLS + EXPECTED_NAMEDTUPLES_SYMBOLS +
            EXPECTED_TRAIN_SYMBOLS))

  def test_estimator_symbols(self):
    symbols = set([name for name, _ in inspect.getmembers(tfgan.estimator)])
    expected_symbols = set(EXPECTED_ESTIMATOR_SYMBOLS)
    self.assertSetEqual(symbols - UNAVOIDABLE_SYMBOLS, expected_symbols)

  def test_eval_symbols(self):
    symbols = set([name for name, _ in inspect.getmembers(tfgan.eval)])
    expected_symbols = set(EXPECTED_EVAL_SYMBOLS)
    self.assertSetEqual(symbols - UNAVOIDABLE_SYMBOLS, expected_symbols)

  def test_features_symbols(self):
    symbols = set([name for name, _ in inspect.getmembers(tfgan.features)])
    expected_symbols = set(EXPECTED_FEATURES_SYMBOLS)
    self.assertSetEqual(symbols - UNAVOIDABLE_SYMBOLS, expected_symbols)

  def test_losses_symbols(self):
    symbols = set([name for name, _ in inspect.getmembers(tfgan.losses)])
    expected_symbols = set(EXPECTED_LOSSES_SYMBOLS)
    self.assertSetEqual(symbols - UNAVOIDABLE_SYMBOLS, expected_symbols)

  def test_losses_wargs_symbols(self):
    symbols = set([name for name, _ in inspect.getmembers(tfgan.losses.wargs)])
    expected_symbols = set(EXPECTED_LOSSES_WARGS_SYMBOLS)
    self.assertSetEqual(symbols - UNAVOIDABLE_SYMBOLS, expected_symbols)


if __name__ == '__main__':
  absltest.main()
