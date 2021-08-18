# coding=utf-8
# Copyright 2021 The TensorFlow GAN Authors.
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

"""Tests for tfgan.examples.esrgan.eval"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import os
import collections

import tensorflow as tf
import eval_lib, networks

HParams = collections.namedtuple('HParams', [
  'num_steps', 'image_dir', 'batch_size', 'num_inception_images',
  'eval_real_images', 'hr_dimension', 'scale', 'trunk_size'])

class EvalTest(tf.test.TestCase):
  def setUp(self):
    self.HParams = HParams(1, '/content/', 
                          2, 2, 
                          True, 256, 
                          4, 11)
    
    d = tf.data.Dataset.from_tensor_slices(tf.random.normal([2, 256, 256, 3]))
    def lr(hr):
      lr = tf.image.resize(hr, [64, 64], method='bicubic')
      return lr, hr

    d = d.map(lr)
    d = d.batch(2)
    self.mock_dataset = d 
    self.generator = networks.generator_network(self.HParams)

  def test_eval(self):
    self.assertIsNone(eval_lib.evaluate(self.HParams, 
                                        self.generator, 
                                        self.mock_dataset))

if __name__ == '__main__':
  tf.test.main()

