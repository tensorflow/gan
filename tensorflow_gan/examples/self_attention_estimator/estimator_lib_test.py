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

"""Tests for estimator_lib.

***NOTE***: It's wise to run this test with `-c opt`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # tf

from tensorflow_gan.examples.self_attention_estimator import estimator_lib
from tensorflow_gan.examples.self_attention_estimator import train_experiment

mock = tf.compat.v1.test.mock


class EstimatorLibTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'use_tpu': False},
      {'use_tpu': True},
  )
  @mock.patch.object(
      estimator_lib.tfgan.eval, 'classifier_score_from_logits', autospec=True)
  @mock.patch.object(
      estimator_lib.tfgan.eval,
      'frechet_classifier_distance_from_activations', autospec=True)
  def test_get_metrics_syntax(self, mock_fid, mock_iscore, use_tpu):
    if tf.executing_eagerly():
      # tf.metrics.mean is not supported when eager execution is enabled.
      return
    bs = 40
    hparams = train_experiment.HParams(
        train_batch_size=1,
        eval_batch_size=bs,
        predict_batch_size=1,
        generator_lr=1.0,
        discriminator_lr=1.0,
        beta1=1.0,
        gf_dim=2,
        df_dim=2,
        num_classes=10,
        shuffle_buffer_size=1,
        z_dim=8,
        model_dir=None,
        max_number_of_steps=None,
        train_steps_per_eval=1,
        num_eval_steps=1,
        debug_params=train_experiment.DebugParams(
            use_tpu=use_tpu,
            eval_on_tpu=use_tpu,
            fake_nets=True,
            fake_data=True,
            continuous_eval_timeout_secs=1,
        ),
        tpu_params=None,
    )

    # Fake arguments to pass to `get_metrics`.
    fake_noise = tf.zeros([bs, 128])
    fake_imgs = tf.zeros([bs, 128, 128, 3])
    fake_lbls = tf.zeros([bs])
    fake_logits = tf.ones([bs, 1008])

    # Mock Inception-inference computations.
    mock_iscore.return_value = 1.0
    mock_fid.return_value = 0.0

    estimator_lib.get_metrics(
        generator_inputs=fake_noise,
        generated_data={'images': fake_imgs, 'labels': fake_lbls},
        real_data={'images': fake_imgs, 'labels': fake_lbls},
        discriminator_real_outputs=(fake_logits, ()),
        discriminator_gen_outputs=(fake_logits, ()),
        hparams=hparams)


if __name__ == '__main__':
  tf.test.main()
