# coding=utf-8
# Copyright 2018 The TensorFlow GAN Authors.
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

"""Simple GAN implementation for arbitrary image generation.

Utilizes TF-GAN GANEstimator along with custom generator and discriminator model
and improved Wasserstein loss to learn image distributions. Set up to run in a
distributed fashion. Requires folder containing relevant images in dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
import tensorflow as tf
import tensorflow_gan as tfgan

layers = tf.contrib.layers


# IO.
flags.DEFINE_string('data_dir', '', 'Dir of image dataset.')
flags.DEFINE_string('save_dir', '', 'Dir to save model checkpoints.')
flags.DEFINE_integer('resize_h', 64, 'Side length to square crop images.')
flags.DEFINE_integer('resize_w', 64, 'Side length to square crop images.')
flags.DEFINE_boolean('add_summaries', True, 'Whether to add default summaries.')

# Batching.
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('seed', 123, 'Random seed for batch loading.')

# Learning.
flags.DEFINE_integer('z_dim', 100, 'Input dimension.')
flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate for adam.')

# Training.
flags.DEFINE_integer('num_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('mod_sum', 100, 'Get summary every mod_sum steps.')
flags.DEFINE_integer('mod_ckpt', 10000, 'Get model ckpt every mod_ckpt steps.')

FLAGS = flags.FLAGS


def open_image(fp, imshape, augment_image=True):
  """Takes filepath and returns a resized tensor with range [-1, 1]."""
  float_im = tf.to_float(tf.image.decode_png(tf.read_file(fp), channels=3))
  expanded_im = tf.expand_dims(float_im, 0)
  resized_im = tf.image.resize_area(expanded_im, imshape, align_corners=True)
  normalized_im = (resized_im - 128.0)/128.0
  if augment_image:
    augmented_im = tf.image.random_flip_left_right(normalized_im)
    return tf.squeeze(augmented_im)
  return tf.squeeze(normalized_im)


def get_get_batch(data_dir, batch_size, z, imshape):
  """Returns function for setting up dataset."""
  def get_batch():
    """Returns batched, repeating, prefetched tensorflow iterator."""
    files = tf.data.Dataset.list_files(data_dir + '/*')
    shuffled_files = files.shuffle(buffer_size=100000)
    image_data = shuffled_files.map(lambda x: open_image(x, imshape),
                                    num_parallel_calls=4)
    prefetched_data = image_data.prefetch(4 * batch_size)
    shuffled_data = prefetched_data.shuffle(buffer_size=batch_size)
    batched_data = shuffled_data.repeat().batch(batch_size, drop_remainder=True)
    data_iterator = batched_data.make_one_shot_iterator()
    noise = tf.random_normal([batch_size, z])
    return noise, data_iterator.get_next()
  return get_batch


leaky = lambda net: tf.nn.leaky_relu(net, 0.2)


def generator(net, mode):
  """DCGAN generator with batchnorm and l2 regularization."""
  with tf.contrib.framework.arg_scope(
      [layers.conv2d_transpose], normalizer_fn=layers.batch_norm,
      normalizer_params={'is_training': (mode == tf.estimator.ModeKeys.TRAIN),
                         'updates_collections': None}):
    net = layers.linear(net, 4 * 4 * 512)
    net = tf.reshape(net, [-1, 4, 4, 512])
    net = layers.conv2d_transpose(net, 256, kernel_size=5, stride=2)
    net = layers.conv2d_transpose(net, 128, kernel_size=5, stride=2)
    net = layers.conv2d_transpose(net, 64, kernel_size=5, stride=2)
    net = layers.conv2d_transpose(net, 3, kernel_size=5, stride=2,
                                  activation_fn=tf.tanh)
    return net


def discriminator(input_net, condition, mode):
  """DCGAN discriminator with batchnorm and l2 regularization."""
  del condition
  with tf.contrib.framework.arg_scope(
      [layers.conv2d], activation_fn=leaky, normalizer_fn=layers.batch_norm,
      normalizer_params={'is_training': (mode == tf.estimator.ModeKeys.TRAIN),
                         'updates_collections': None}):

    net = layers.conv2d(input_net, 64, 5, stride=1, normalizer_fn=None)
    net = layers.conv2d(net, 128, 5, stride=2)
    net = layers.conv2d(net, 256, 5, stride=2)
    net = layers.conv2d(net, 512, 5, stride=2)
    return layers.linear(net, 1, activation_fn=tf.nn.sigmoid)


def _d_loss(gan_model, add_summaries):
  """Improved Wasserstein loss (i.e. with gradient penalty) and summaries.

  Discriminator histograms and gradient summaries are defined here. This is
  necessary because the tfgan abstraction makes it difficult to access both loss
  and variables in a different location.

  Args:
    gan_model: named gan model tuple containing relevant information for losses
      and summaries.
    add_summaries: adds Wasserstein summaries.

  Returns:
    Loss for discriminator.
  """
  loss = tfgan.losses.wasserstein_discriminator_loss(
      gan_model, add_summaries=add_summaries)

  grad_pen = 10 * tfgan.losses.wasserstein_gradient_penalty(
      gan_model, add_summaries=add_summaries)
  return loss + grad_pen


def get_model(runconf=None, add_summaries=True):
  """Returns a gan estimator model."""
  summary_types = None
  if add_summaries:
    summary_types = [tfgan.estimator.SummaryType.IMAGES,
                     tfgan.estimator.SummaryType.VARIABLES]

  gan_estimator = tfgan.estimator.GANEstimator(
      generator_fn=generator,
      discriminator_fn=discriminator,
      generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
      discriminator_loss_fn=_d_loss,
      generator_optimizer=tf.contrib.estimator.clip_gradients_by_norm(
          tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=0.5), 2),
      discriminator_optimizer=tf.contrib.estimator.clip_gradients_by_norm(
          tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=0.5), 2),
      add_summaries=summary_types,
      config=runconf
  )

  return gan_estimator


def main(_):
  imshape = (FLAGS.resize_h, FLAGS.resize_w)
  runconf = tf.estimator.RunConfig(model_dir=FLAGS.save_dir,
                                   tf_random_seed=FLAGS.seed,
                                   save_summary_steps=FLAGS.mod_sum,
                                   save_checkpoints_steps=FLAGS.mod_ckpt,
                                   log_step_count_steps=FLAGS.mod_sum)

  gan_estimator = get_model(runconf, FLAGS.add_summaries)
  input_fn = get_get_batch(
      FLAGS.data_dir, FLAGS.batch_size, FLAGS.z_dim, imshape)
  train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn, max_steps=FLAGS.num_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)

  tf.estimator.train_and_evaluate(gan_estimator, train_spec, eval_spec)


if __name__ == '__main__':
  tf.app.run(main)
