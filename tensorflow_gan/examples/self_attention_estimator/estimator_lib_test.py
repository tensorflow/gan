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
import numpy as np
import tensorflow as tf  # tf

from tensorflow_gan.examples.self_attention_estimator import estimator_lib
from tensorflow_gan.examples.self_attention_estimator import train_experiment

mock = tf.compat.v1.test.mock


def generator(inputs):
  gvar = tf.compat.v1.get_variable('dummy_g', initializer=2.0)
  return gvar * inputs


def discriminator(inputs, _):
  if isinstance(inputs, dict):
    inputs = inputs['images']
  net = tf.math.reduce_sum(inputs, axis=1)
  return tf.compat.v1.get_variable('dummy_d', initializer=2.0) * net


def input_fn(params):
  bs = params['batch_size']
  dummy_imgs = np.zeros([32, 128, 128, 3], dtype=np.float32)
  ds = tf.data.Dataset.from_tensor_slices(dummy_imgs)
  def _set_shape(x):
    x.set_shape([bs, None, None, None])
    return x
  ds = ds.batch(bs).map(_set_shape).map(lambda x: (x, x)).repeat()
  return ds


def _new_tensor(*args, **kwargs):
  del args, kwargs
  # Tensors need to be created in the same graph, so generate them at the call
  # site.
  return (tf.ones([32, 128, 3]), tf.ones([32, 128, 3]))


class EstimatorLibTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(EstimatorLibTest, self).setUp()
    self.hparams = train_experiment.HParams(
        train_batch_size=1,
        eval_batch_size=32,
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
            use_tpu=False,
            eval_on_tpu=False,
            fake_nets=None,
            fake_data=None,
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


  @mock.patch.object(
      estimator_lib.tfgan.eval,
      'classifier_score_from_logits_streaming', new=_new_tensor)
  @mock.patch.object(
      estimator_lib.tfgan.eval,
      'frechet_classifier_distance_from_activations_streaming', new=_new_tensor)
  @mock.patch.object(estimator_lib.eval_lib, 'get_activations', new=_new_tensor)
  def test_get_gpu_estimator_syntax(self):
    config = estimator_lib.get_run_config_from_hparams(self.hparams)
    est = estimator_lib.get_gpu_estimator(
        generator, discriminator, self.hparams, config)
    est.evaluate(lambda: input_fn({'batch_size': 16}), steps=1)


if __name__ == '__main__':
  tf.test.main()
