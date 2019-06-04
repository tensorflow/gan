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

"""Trains a GANEstimator on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tensorflow_gan.examples.mnist_estimator import train_lib

flags.DEFINE_integer('batch_size', 32,
                     'The number of images in each train batch.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'noise_dims', 64, 'Dimensions of the generator noise vector')

flags.DEFINE_string('output_dir', '/tmp/tfgan_logdir/mnist-estimator/',
                    'Directory where the results are saved to.')

FLAGS = flags.FLAGS


def main(_):
  hparams = train_lib.HParams(FLAGS.batch_size, FLAGS.max_number_of_steps,
                              FLAGS.noise_dims, FLAGS.output_dir)
  train_lib.train(hparams)


if __name__ == '__main__':
  app.run(main)
