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

"""Evaluates a conditional TF-GAN trained MNIST model."""

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

flags.DEFINE_string('conditional_eval_checkpoint_dir_mnist', '/tmp/mnist/',
                    'Directory where the model was written to.')

flags.DEFINE_string('conditional_eval_dir_mnist', '/tmp/mnist/',
                    'Directory where the results are saved to.')

flags.DEFINE_integer('num_images_per_class', 10,
                     'Number of images to generate per class.')

flags.DEFINE_integer('noise_dims_mnist_eval', 64,
                     'Dimensions of the generator noise vector')

flags.DEFINE_string(
    'classifier_filename_cond_eval', None,
    'Location of the pretrained classifier. If `None`, use '
    'default.')

flags.DEFINE_integer(
    'max_number_of_evaluations_mnist_cond_eval', None,
    'Number of times to run evaluation. If `None`, run '
    'forever.')

flags.DEFINE_boolean('write_to_disk_mnist_cond_eval', True, 'If `True`, run images to disk.')

FLAGS = flags.FLAGS
NUM_CLASSES = 10


def main(_, run_eval_loop=True):
  with tf.compat.v1.name_scope('inputs'):
    noise, one_hot_labels = _get_generator_inputs(FLAGS.num_images_per_class,
                                                  NUM_CLASSES, FLAGS.noise_dims_mnist_eval)

  # Generate images.
  with tf.compat.v1.variable_scope('Generator'):  # Same scope as in train job.
    images = networks.conditional_generator((noise, one_hot_labels),
                                            is_training=False)

  # Visualize images.
  reshaped_img = tfgan.eval.image_reshaper(
      images, num_cols=FLAGS.num_images_per_class)
  tf.compat.v1.summary.image('generated_images', reshaped_img, max_outputs=1)

  # Calculate evaluation metrics.
  tf.compat.v1.summary.scalar(
      'MNIST_Classifier_score',
      util.mnist_score(images, FLAGS.classifier_filename_cond_eval))
  tf.compat.v1.summary.scalar(
      'MNIST_Cross_entropy',
      util.mnist_cross_entropy(images, one_hot_labels,
                               FLAGS.classifier_filename_cond_eval))

  # Write images to disk.
  image_write_ops = None
  if FLAGS.write_to_disk_mnist_cond_eval:
    image_write_ops = tf.io.write_file(
        '%s/%s' % (FLAGS.conditional_eval_dir_mnist, 'conditional_gan.png'),
        tf.image.encode_png(
            data_provider.float_image_to_uint8(reshaped_img[0])))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop:
    return
  evaluation.evaluate_repeatedly(
      FLAGS.conditional_eval_checkpoint_dir_mnist,
      hooks=[
          evaluation.SummaryAtEndHook(FLAGS.conditional_eval_dir_mnist),
          evaluation.StopAfterNEvalsHook(1)
      ],
      eval_ops=image_write_ops,
      max_number_of_evaluations_mnist_cond_eval=FLAGS.max_number_of_evaluations_mnist_cond_eval)


def _get_generator_inputs(num_images_per_class, num_classes, noise_dims_mnist_eval):
  """Return generator inputs for evaluation."""
  # Since we want a grid of numbers for the conditional generator, manually
  # construct the desired class labels.
  num_images_generated_mnist = num_images_per_class * num_classes
  noise = tf.random.normal([num_images_generated_mnist, noise_dims_mnist_eval])
  # pylint:disable=g-complex-comprehension
  labels = [
      lbl for lbl in range(num_classes) for _ in range(num_images_per_class)
  ]
  one_hot_labels = tf.one_hot(tf.constant(labels), num_classes)
  return noise, one_hot_labels


if __name__ == '__main__':
  app.run(main)
