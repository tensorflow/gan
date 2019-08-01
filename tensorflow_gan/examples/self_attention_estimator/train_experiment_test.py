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

"""Tests for mnist_estimator_tpu.train_experiment.

***NOTE***: It's wise to run this test with `-c opt`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf  # tf

from tensorflow_gan.examples.self_attention_estimator import train_experiment

FLAGS = flags.FLAGS
mock = tf.compat.v1.test.mock


def _get_real_activations_mock(*args, **kwargs):
  del args, kwargs  # Unused.
  # Used to mock out an expensive read operation.
  # Note that the standard autospec=True will not work in this case since we
  # need the generated tensors to be created at the call site.
  # TODO(dyoel): Remove once b/135294174 is resolved.
  return tf.zeros([16, 1008]), tf.zeros([16, 2048])


class TrainExperimentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TrainExperimentTest, self).setUp()
    # Make the test on CPU.
    FLAGS.use_tpu = False
    FLAGS.eval_on_tpu = False

    # Make small batch sizes.
    FLAGS.z_dim = 4
    FLAGS.train_batch_size = 4
    FLAGS.eval_batch_size = 16
    FLAGS.fake_data = True
    FLAGS.fake_nets = True

    # Make small networks.
    FLAGS.gf_dim = 2
    FLAGS.df_dim = 4

    # Take few steps.
    FLAGS.max_number_of_steps = 1
    FLAGS.num_eval_steps = 1
    FLAGS.continuous_eval_timeout_secs = 1
    FLAGS.tpu_iterations_per_loop = 1

  @parameterized.parameters(
      {'mode': 'train', 'tpu_est': True},
      {'mode': 'train', 'tpu_est': False},
      {'mode': 'continuous_eval', 'tpu_est': True},
      {'mode': 'continuous_eval', 'tpu_est': False},
      {'mode': 'train_and_eval', 'tpu_est': True},
  )
  @mock.patch.object(train_experiment.est_lib, 'get_metrics', autospec=True)
  def test_cpu_local(self, mock_metrics, mode, tpu_est):
    """Tests the flag configuration for training on CPU."""
    FLAGS.mode = mode
    FLAGS.model_dir = self.create_tempdir().full_path
    FLAGS.session_master = None
    FLAGS.use_tpu_estimator = tpu_est

    # Mock computationally expensive metrics computations.
    mock_metrics.return_value = {}

    train_experiment.main(None)

  @parameterized.parameters(
      {'mode': tf.estimator.ModeKeys.TRAIN, 'tpu_est': True},
      {'mode': tf.estimator.ModeKeys.EVAL, 'tpu_est': True},
      {'mode': tf.estimator.ModeKeys.TRAIN, 'tpu_est': False},
      {'mode': tf.estimator.ModeKeys.EVAL, 'tpu_est': False},
  )
  @flagsaver.flagsaver
  def test_input_fn(self, mode, tpu_est):
    """Tests input_fn."""
    FLAGS.fake_data = False
    FLAGS.use_tpu_estimator = tpu_est
    params = {
        'batch_size': 8,
        'z_dim': 12,
        'shuffle_buffer_size': 100,
    }
    train_experiment.train_eval_input_fn(mode, params)

  def test_make_estimator(self):
    train_experiment.make_estimator(
        train_experiment.HParams(
            train_batch_size=4,
            eval_batch_size=4,
            predict_batch_size=4,
            use_tpu=False,
            eval_on_tpu=False,
            generator_lr=0.1,
            discriminator_lr=0.1,
            beta1=0.9,
            gf_dim=2,
            df_dim=4,
            num_classes=1000,
            shuffle_buffer_size=10000,
            z_dim=4,
        ))

  @parameterized.parameters(
      {'tpu_est': True},
      {'tpu_est': False},
  )
  @mock.patch.object(
      train_experiment.est_lib.eval_lib,
      'get_real_activations',
      new=_get_real_activations_mock)
  def test_estimator_train(self, tpu_est):
    # TODO(dyoel): Enable eval_on_tpu once b/135294174 is resolved.
    FLAGS.model_dir = self.create_tempdir().full_path
    FLAGS.use_tpu_estimator = tpu_est
    FLAGS.train_batch_size = 4
    FLAGS.eval_batch_size = 32

    # Note that make_estimator reads the batch sizes from flags, so we must use
    # the same values.
    hparams = train_experiment.HParams(
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=4,
        use_tpu=tpu_est,
        eval_on_tpu=False,
        generator_lr=0.1,
        discriminator_lr=0.1,
        beta1=0.9,
        gf_dim=2,
        df_dim=4,
        num_classes=1000,
        shuffle_buffer_size=10000,
        z_dim=FLAGS.z_dim,
    )
    estimator = train_experiment.make_estimator(hparams)
    estimator.train(train_experiment.train_eval_input_fn, steps=1)


if __name__ == '__main__':
  tf.test.main()
