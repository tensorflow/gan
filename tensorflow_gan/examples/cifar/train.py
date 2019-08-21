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

"""Trains a generator on CIFAR data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from tensorflow_gan.examples.cifar import train_lib

# ML Hparams.
flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')
flags.DEFINE_integer('max_number_of_steps', 1000000,
                     'The maximum number of gradient steps.')
flags.DEFINE_float('generator_lr', 0.0002, 'The generator learning rate.')
flags.DEFINE_float('discriminator_lr', 0.0002,
                   'The discriminator learning rate.')

# ML Infrastructure.
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_string('train_log_dir', '/tmp/tfgan_logdir/cifar/',
                    'Directory where to write event logs.')
flags.DEFINE_integer(
    'ps_replicas', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

FLAGS = flags.FLAGS


def main(_):
  hparams = train_lib.HParams(FLAGS.batch_size, FLAGS.max_number_of_steps,
                              FLAGS.generator_lr, FLAGS.discriminator_lr,
                              FLAGS.master, FLAGS.train_log_dir,
                              FLAGS.ps_replicas, FLAGS.task)
  train_lib.train(hparams)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
