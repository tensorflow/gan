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

"""Tests for CIFAR10 eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_gan.examples.cifar import eval_lib

mock = tf.compat.v1.test.mock


class EvalTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'eval_real_images': True},
      {
          'eval_real_images': False,
      },
  )
  @mock.patch.object(
      eval_lib.util, 'get_frechet_inception_distance', autospec=True)
  @mock.patch.object(eval_lib.util, 'get_inception_scores', autospec=True)
  @mock.patch.object(eval_lib.data_provider, 'provide_data', autospec=True)
  def test_build_graph(self, mock_provide_data, mock_iscore, mock_fid,
                       eval_real_images):
    hparams = eval_lib.HParams(
        master='',
        checkpoint_dir='/tmp/cifar10/',
        eval_dir='/tmp/cifar10/',
        num_images_generated=100,
        num_inception_images=10,
        eval_real_images=eval_real_images,
        eval_frechet_inception_distance=True,
        max_number_of_evaluations=None,
        write_to_disk=True)

    # Mock reads from disk.
    mock_provide_data.return_value = (tf.ones(
        [hparams.num_images_generated, 32, 32,
         3]), tf.zeros([hparams.num_images_generated]))

    # Mock `frechet_inception_distance` and `inception_score`, which are
    # expensive.
    mock_fid.return_value = 1.0
    mock_iscore.return_value = 1.0

    eval_lib.evaluate(hparams, run_eval_loop=False)


if __name__ == '__main__':
  tf.test.main()
