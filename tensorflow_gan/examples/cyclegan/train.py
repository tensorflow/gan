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

"""Trains a CycleGAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tensorflow_gan.examples.cyclegan import train_lib

flags.DEFINE_string('image_set_x_file_pattern', None,
                    'File pattern of images in image set X')
flags.DEFINE_string('image_set_y_file_pattern', None,
                    'File pattern of images in image set Y')
flags.DEFINE_integer('batch_size', 1, 'The number of images in each batch.')
flags.DEFINE_integer('patch_size', 64, 'The patch size of images.')
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_string('train_log_dir', '/tmp/tfgan_logdir/cyclegan/',
                    'Directory where to write event logs.')
flags.DEFINE_float('generator_lr', 0.0002,
                   'The compression model learning rate.')
flags.DEFINE_float('discriminator_lr', 0.0001,
                   'The discriminator learning rate.')
flags.DEFINE_integer('max_number_of_steps', 500000,
                     'The maximum number of gradient steps.')
flags.DEFINE_integer(
    'ps_replicas', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')
flags.DEFINE_float('cycle_consistency_loss_weight', 10.0,
                   'The weight of cycle consistency loss')

FLAGS = flags.FLAGS


def main(_):
  hparams = train_lib.HParams(
      FLAGS.image_set_x_file_pattern, FLAGS.image_set_y_file_pattern,
      FLAGS.batch_size, FLAGS.patch_size, FLAGS.master, FLAGS.train_log_dir,
      FLAGS.generator_lr, FLAGS.discriminator_lr, FLAGS.max_number_of_steps,
      FLAGS.ps_replicas, FLAGS.task, FLAGS.cycle_consistency_loss_weight)
  train_lib.train(hparams)


if __name__ == '__main__':
  app.run(main)
