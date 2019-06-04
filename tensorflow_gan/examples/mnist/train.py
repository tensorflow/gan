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

"""Trains a generator on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from tensorflow_gan.examples.mnist import train_lib


flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/tfgan_logdir/mnist',
                    'Directory where to write event logs.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_string('gan_type', 'unconditional',
                    'Either `unconditional`, `conditional`, or `infogan`.')

flags.DEFINE_integer('grid_size', 5, 'Grid size for image visualization.')

flags.DEFINE_integer('noise_dims', 64,
                     'Dimensions of the generator noise vector.')

FLAGS = flags.FLAGS


def main(_):
  hparams = train_lib.HParams(FLAGS.batch_size, FLAGS.train_log_dir,
                              FLAGS.max_number_of_steps, FLAGS.gan_type,
                              FLAGS.grid_size, FLAGS.noise_dims)
  train_lib.train(hparams)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
