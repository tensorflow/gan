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

"""Tests for train_experiment.

***NOTE***: It's wise to run this test as an optimized binary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # tf

from tensorflow_gan.examples.self_attention_estimator import train_experiment

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
    self.hparams = train_experiment.HParams(
        # Make small batch sizes.
        z_dim=4,
        train_batch_size=4,
        eval_batch_size=16,
        predict_batch_size=1,

        # Make small networks.
        gf_dim=2,
        df_dim=4,

        # Take few steps.
        max_number_of_steps=1,
        num_eval_steps=1,
        model_dir=self.create_tempdir().full_path,
        train_steps_per_eval=1,
        generator_lr=1.0,
        discriminator_lr=1.0,
        beta1=1.0,
        shuffle_buffer_size=1,
        num_classes=10,
        debug_params=train_experiment.DebugParams(
            # Make the test on CPU.
            use_tpu=False,
            eval_on_tpu=False,
            fake_data=True,
            fake_nets=True,
            continuous_eval_timeout_secs=1,
        ),
        tpu_params=train_experiment.TPUParams(
            use_tpu_estimator=False,
            tpu_location='local',
            gcp_project=None,
            tpu_zone=None,
            tpu_iterations_per_loop=1,
        ),
    )

  def test_run_train_cpu_local_gpuestimator(self):
    """Tests `run_train`."""
    train_experiment.run_train(self.hparams)


  @mock.patch.object(train_experiment.est_lib, 'get_metrics', autospec=True)
  def test_run_continuous_eval_cpu_local_gpuestimator(self, _):
    """Tests `run_continuous_eval`."""
    train_experiment.run_continuous_eval(self.hparams)


  @mock.patch.object(train_experiment.est_lib, 'get_metrics', autospec=True)
  def test_train_and_eval_cpu_local(self, mock_metrics):
    """Tests `run_train_and_eval`."""
    # Mock computationally expensive metrics computations.
    mock_metrics.return_value = {}
    train_experiment.run_train_and_eval(self.hparams)

  @parameterized.parameters(
      {'mode': tf.estimator.ModeKeys.TRAIN, 'tpu_est': True},
      {'mode': tf.estimator.ModeKeys.EVAL, 'tpu_est': True},
      {'mode': tf.estimator.ModeKeys.TRAIN, 'tpu_est': False},
      {'mode': tf.estimator.ModeKeys.EVAL, 'tpu_est': False},
  )
  @mock.patch.object(
      train_experiment.data_provider, 'provide_dataset', autospec=True)
  def test_input_fn(self, mock_dataset, mode, tpu_est):
    """Tests input_fn."""
    params = {
        'tpu_params': self.hparams.tpu_params._replace(
            use_tpu_estimator=tpu_est),
        'batch_size': 8,
        'train_batch_size': 8,
        'eval_batch_size': 8,
        'predict_batch_size': 16,
        'debug_params': self.hparams.debug_params._replace(fake_data=False),
        'z_dim': 12,
        'shuffle_buffer_size': 100,
    }
    mock_dataset.return_value = tf.data.Dataset.from_tensors(
        np.zeros([8, 128, 128, 3])).map(lambda x: (x, [1]))
    train_experiment.train_eval_input_fn(mode, params)

  def test_make_estimator(self):
    train_experiment.make_estimator(self.hparams)

  @mock.patch.object(
      train_experiment.est_lib.eval_lib,
      'get_real_activations',
      new=_get_real_activations_mock)
  def test_estimator_train_notpu(self):
    estimator = train_experiment.make_estimator(self.hparams)
    estimator.train(train_experiment.train_eval_input_fn, steps=1)



if __name__ == '__main__':
  tf.test.main()
