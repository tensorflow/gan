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

"""Evaluates a TF-GAN trained MNIST model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tensorflow_gan.examples.mnist import eval_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', '/tmp/mnist/',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/tmp/mnist/',
                    'Directory where the results are saved to.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_integer('num_images_generated', 1000,
                     'Number of images to generate at once.')

flags.DEFINE_boolean('eval_real_images', False,
                     'If `True`, run Inception network on real images.')

flags.DEFINE_integer('noise_dims', 64,
                     'Dimensions of the generator noise vector')

flags.DEFINE_integer(
    'max_number_of_evaluations', None,
    'Number of times to run evaluation. If `None`, run '
    'forever.')

flags.DEFINE_boolean('write_to_disk', True, 'If `True`, run images to disk.')


def main(_):
  hparams = eval_lib.HParams(FLAGS.checkpoint_dir, FLAGS.eval_dir,
                             FLAGS.dataset_dir, FLAGS.num_images_generated,
                             FLAGS.eval_real_images, FLAGS.noise_dims,
                             FLAGS.max_number_of_evaluations,
                             FLAGS.write_to_disk)
  eval_lib.evaluate(hparams, run_eval_loop=True)


if __name__ == '__main__':
  app.run(main)
