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

"""Tests for eval_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_gan.examples.self_attention_estimator import eval_lib

mock = tf.compat.v1.test.mock


def _mock_inception(*args, **kwargs):  # pylint: disable=function-redefined
  del args, kwargs
  return {
      'logits': tf.zeros([12, 1008]),
      'pool_3': tf.zeros([12, 2048]),
  }


class EvalLibTest(tf.test.TestCase):

  @mock.patch.object(eval_lib.tfgan.eval, 'sample_and_run_inception',
                     new=_mock_inception)
  @mock.patch.object(eval_lib.data_provider, 'provide_dataset', autospec=True)
  def test_get_real_activations_syntax(self, mock_dataset):
    mock_dataset.return_value = tf.data.Dataset.from_tensors(
        np.zeros([128, 128, 3])).map(lambda x: (x, 1))
    real_pools = eval_lib.get_real_activations(
        batch_size=4, num_batches=3)
    real_pools.shape.assert_is_compatible_with([4 * 3, 2048])


if __name__ == '__main__':
  tf.test.main()
