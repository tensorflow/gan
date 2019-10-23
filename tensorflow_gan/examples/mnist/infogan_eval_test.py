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

"""Tests for tfgan.examples.mnist.infogan_eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_gan.examples.mnist import infogan_eval_lib


class MnistInfoGANEvalTest(tf.test.TestCase):

  def test_build_graph(self):
    if tf.executing_eagerly():
      return
    hparams = infogan_eval_lib.HParams(
        checkpoint_dir='/tmp/mnist/',
        eval_dir='/tmp/mnist/',
        noise_samples=6,
        unstructured_noise_dims=62,
        continuous_noise_dims=2,
        max_number_of_evaluations=None,
        write_to_disk=True)
    infogan_eval_lib.evaluate(hparams, run_eval_loop=False)


if __name__ == '__main__':
  tf.test.main()
