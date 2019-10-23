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

"""Evaluates an InfoGAN TFGAN trained MNIST model.

The image visualizations, as in https://arxiv.org/abs/1606.03657, show the
effect of varying a specific latent variable on the image. Each visualization
focuses on one of the three structured variables. Columns have two of the three
variables fixed, while the third one is varied. Different rows have different
random samples from the remaining latents.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tensorflow_gan.examples.mnist import infogan_eval_lib

flags.DEFINE_string('checkpoint_dir', '/tmp/mnist/',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/tmp/mnist/',
                    'Directory where the results are saved to.')

flags.DEFINE_integer(
    'noise_samples', 6,
    'Number of samples to draw from the continuous structured '
    'noise.')

flags.DEFINE_integer('unstructured_noise_dims', 62,
                     'The number of dimensions of the unstructured noise.')

flags.DEFINE_integer('continuous_noise_dims', 2,
                     'The number of dimensions of the continuous noise.')

flags.DEFINE_integer(
    'max_number_of_evaluations', None,
    'Number of times to run evaluation. If `None`, run '
    'forever.')

flags.DEFINE_boolean('write_to_disk', True, 'If `True`, run images to disk.')

FLAGS = flags.FLAGS


def main(_):
  hparams = infogan_eval_lib.HParams(
      FLAGS.checkpoint_dir, FLAGS.eval_dir, FLAGS.noise_samples,
      FLAGS.unstructured_noise_dims, FLAGS.continuous_noise_dims,
      FLAGS.max_number_of_evaluations,
      FLAGS.write_to_disk)
  infogan_eval_lib.evaluate(hparams, run_eval_loop=True)


if __name__ == '__main__':
  app.run(main)
