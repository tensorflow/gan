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

"""Tests for tfgan.examples.mnist.eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_gan.examples.mnist import eval_lib

mock = tf.compat.v1.test.mock


class EvalTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('RealData', True), ('GeneratedData', False))
  @mock.patch.object(eval_lib.data_provider, 'provide_data', autospec=True)
  @mock.patch.object(eval_lib, 'util', autospec=True)
  def test_build_graph(self, eval_real_images, mock_util, mock_provide_data):
    hparams = eval_lib.HParams(
        checkpoint_dir='/tmp/mnist/',
        eval_dir='/tmp/mnist/',
        dataset_dir=None,
        num_images_generated=1000,
        eval_real_images=eval_real_images,
        noise_dims=64,
        max_number_of_evaluations=None,
        write_to_disk=True)

    # Mock input pipeline.
    bs = hparams.num_images_generated
    mock_imgs = np.zeros([bs, 28, 28, 1], dtype=np.float32)
    mock_lbls = np.concatenate((np.ones([bs, 1], dtype=np.int32),
                                np.zeros([bs, 9], dtype=np.int32)),
                               axis=1)
    mock_provide_data.return_value = (mock_imgs, mock_lbls)

    # Mock expensive eval metrics.
    mock_util.mnist_frechet_distance.return_value = 1.0
    mock_util.mnist_score.return_value = 0.0

    eval_lib.evaluate(hparams, run_eval_loop=False)


if __name__ == '__main__':
  tf.test.main()
