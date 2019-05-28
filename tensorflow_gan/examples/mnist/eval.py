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

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples import evaluation_helper as evaluation
from tensorflow_gan.examples.mnist import data_provider
from tensorflow_gan.examples.mnist import networks
from tensorflow_gan.examples.mnist import util

FLAGS = flags.FLAGS

flags.DEFINE_string('eval_checkpoint_dir_mnist', '/tmp/mnist/',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_eval_dir_mnist', '/tmp/mnist/',
                    'Directory where the results are saved to.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_integer('num_images_generated_mnist', 1000,
                     'Number of images to generate at once.')

flags.DEFINE_boolean('eval_real_images_mnist', False,
                     'If `True`, run Inception network on real images.')

flags.DEFINE_integer('noise_dims', 64,
                     'Dimensions of the generator noise vector')

flags.DEFINE_string(
    'classifier_filename_eval', None,
    'Location of the pretrained classifier. If `None`, use '
    'default.')

flags.DEFINE_integer(
    'max_number_of_evaluations_mnist_eval', None,
    'Number of times to run evaluation. If `None`, run '
    'forever.')

flags.DEFINE_boolean('write_to_disk_mnist_eval', True, 'If `True`, run images to disk.')


def main(_, run_eval_loop=True):
  # Fetch real images.
  with tf.compat.v1.name_scope('inputs'):
    real_images, _ = data_provider.provide_data(
        'train', FLAGS.num_images_generated_mnist, FLAGS.dataset_dir)

  image_write_ops = None
  if FLAGS.eval_real_images_mnist:
    tf.compat.v1.summary.scalar(
        'MNIST_Classifier_score',
        util.mnist_score(real_images, FLAGS.classifier_filename_eval))
  else:
    # In order for variables to load, use the same variable scope as in the
    # train job.
    with tf.compat.v1.variable_scope('Generator'):
      images = networks.unconditional_generator(
          tf.random.normal([FLAGS.num_images_generated_mnist, FLAGS.noise_dims]),
          is_training=False)
    tf.compat.v1.summary.scalar(
        'MNIST_Frechet_distance',
        util.mnist_frechet_distance(real_images, images,
                                    FLAGS.classifier_filename_eval))
    tf.compat.v1.summary.scalar(
        'MNIST_Classifier_score',
        util.mnist_score(images, FLAGS.classifier_filename_eval))
    if FLAGS.num_images_generated_mnist >= 100 and FLAGS.write_to_disk_mnist_eval:
      reshaped_images = tfgan.eval.image_reshaper(
          images[:100, ...], num_cols=10)
      uint8_images = data_provider.float_image_to_uint8(reshaped_images)
      image_write_ops = tf.io.write_file(
          '%s/%s' % (FLAGS.eval_eval_dir_mnist, 'unconditional_gan.png'),
          tf.image.encode_png(uint8_images[0]))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop:
    return
  evaluation.evaluate_repeatedly(
      FLAGS.eval_checkpoint_dir_mnist,
      hooks=[
          evaluation.SummaryAtEndHook(FLAGS.eval_eval_dir_mnist),
          evaluation.StopAfterNEvalsHook(1)
      ],
      eval_ops=image_write_ops,
      max_number_of_evaluations_mnist_eval=FLAGS.max_number_of_evaluations_mnist_eval)


if __name__ == '__main__':
  app.run(main)
