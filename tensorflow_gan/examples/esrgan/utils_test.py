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

"""Tests for tfgan.examples.esrgan.utils"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import os
import collections

import tensorflow as tf
import utils, networks


Params = collections.namedtuple('HParams', ['hr_dimension', 
                                            'scale', 
                                            'trunk_size',
                                            'path'])

class UtilsTest(tf.test.TestCase):
  def setUp(self):
    super(UtilsTest, self).setUp()
    self.HParams = Params(256, 4, 11, '/content/')
    
    self.generator1 = networks.generator_network(self.HParams)
    self.generator1.save(self.HParams.path+'1/')
    self.generator2 = networks.generator_network(self.HParams)
    self.generator1.save(self.HParams.path+'2/')

    self.hr_data = tf.random.normal([2,256,256,3])
    self.lr_data = tf.random.normal([2,64,64,3])
    self.gen_data = tf.random.normal([2,256,256,3])
    
  def test_visualize_results(self):
    """ To test display grid function. The function doesn't return anything if no 
        error is found."""
    self.assertIsNone(utils.visualize_results(self.lr_data,
                                              self.gen_data,
                                              self.hr_data))
  def test_psnr(self):
    psnr = utils.get_psnr(self.hr_data, self.gen_data)
    self.assertEqual(psnr.dtype, tf.float32)
  
  def test_fid_score(self):
    fid_value = utils.get_frechet_inception_distance(self.hr_data,
                                                     self.gen_data,
                                                     batch_size=2, 
                                                     num_inception_images=2)
    self.assertEqual(fid_value.dtype, tf.float32)

  def test_inception_score(self):
    is_score = utils.get_inception_scores(self.gen_data,
                                          batch_size=2, 
                                          num_inception_images=2)
    self.assertEqual(is_score.dtype, tf.float32)
  
  def test_interpolation(self):
    """ To test the interpolation function. """
    inter_gen = utils.network_interpolation(phase_1_path=self.HParams.path+'1/',
                                            phase_2_path=self.HParams.path+'2/')
    self.assertEqual(type(inter_gen), type(self.generator1))

if __name__ == '__main__':
  tf.test.main()
