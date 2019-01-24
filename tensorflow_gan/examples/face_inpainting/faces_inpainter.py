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

"""Implements Semantic Image Inpainting With Deep Generative Models.

This file implements an inpainting estimator that wraps around a trained
GANEstimator. The inpainting estimator trains a single variable z, representing
the hidden latent distribution that is the 'noise' input to the GANEstimator.
By training z, the inpainting estimator can move around the latent z space
towards minimizing a specific loss function. In this example, we utilize z space
traversal to find the appropriate z vector to inpaint a masked image.

See https://arxiv.org/abs/1607.07539 for more.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from absl import flags
import numpy as np
from scipy.signal import convolve2d
import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan.examples.face_inpainting import faces_gan

flags.DEFINE_integer('mask_side_len', 20, 'Side length of the mask used for '
                     'inference.')
flags.DEFINE_boolean('warmstart', True, 'Whether or not to reload an '
                     'underlying GAN from checkpoint.')

FLAGS = flags.FLAGS


def _create_weighted_mask(mask, kernel_size=7):
  """Creates a weighted mask based on the input mask.

  Creates weight matrix of the same shape as mask, that weights pixels near the
  input mask edge lower. The weights of each pixel are calculated by convolving
  a kernel of size kernel_size over the input mask. Pixel weights are calculated
  as the fraction of masked pixels in the kernel over the total number of
  pixels in the kernel. This weighted mask implements an inverted version of the
  proposal in the paper (i.e. pixels close to the mask are weighted _lower_),
  because this is easier to implement with a simple convolve. The actual mask
  can then be inverted. See https://arxiv.org/abs/1607.07539 for more.

  Args:
    mask: the input mask to convolve over. Output weights are based on this
      mask.
    kernel_size: the size of the kernel used to convolve. A larger number
      results in more dispersed weights.
  Returns:
    A weight matrix that weights pixels closest to the input mask lower.
  """
  ker = np.ones((kernel_size, kernel_size), dtype=np.float32)
  ker = ker / np.sum(ker)
  return mask * convolve2d(mask, ker, mode='same', boundary='symm')


def get_input_fn(data_dir, batch_size, imshape, mask_side_len):
  """Returns the inpainting input function.

  In this simple example, masked components are squares with given side length.
  More complex shapes are possible, including a free form 'drawn' mask input.
  Note that the input image and the mask must match the size of the underlying
  GANEstimator that is being used, as the GANEstimator is not necessarily
  resolution independent.

  Args:
    data_dir: string representing the path to a folder of input images.
    batch_size: integer representing the number of images to test at once.
    imshape: the size to shape the input to. Should match the size the
      GANEstimator was trained on.
    mask_side_len: integer representing the side length of a square mask.
  Returns:
    An input function that defines a tf.Dataset object with the above
    properties.
  """

  # Generate the mask outside the actual input function so that it does not
  # change while training the z input.
  if imshape[0] <= mask_side_len or imshape[1] <= mask_side_len:
    raise ValueError('Image shape is too small compared to mask side length.')
  mask = np.ones(shape=imshape)
  mask_y = random.randint(0, imshape[0] - mask_side_len)
  mask_x = random.randint(0, imshape[1] - mask_side_len)
  mask[mask_y:mask_y + mask_side_len, mask_x: mask_x + mask_side_len] = 0.0
  weighted_mask = _create_weighted_mask(mask)

  def input_fn():
    """Feeds masks and images to be inpainted to the estimator."""
    files = tf.data.Dataset.list_files(data_dir + '/*', shuffle=False)
    files = files.take(batch_size)
    data = files.map(lambda x: faces_gan.open_image(x, imshape, False),
                     num_parallel_calls=4)
    data = data.cache().repeat().batch(
        batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    image_iterator = data.make_one_shot_iterator()
    expanded_weighted_mask = tf.expand_dims(
        tf.convert_to_tensor(weighted_mask, dtype=tf.float32), -1)
    expanded_mask = tf.expand_dims(
        tf.convert_to_tensor(mask, dtype=tf.float32), -1)
    features = {
        'mask': expanded_mask,
        'weighted_mask': expanded_weighted_mask
    }
    return features, image_iterator.get_next()
  return input_fn


def inpainter_loss(gan_model, features, labels, add_summaries):
  """Loss for inpainter, taken from https://arxiv.org/abs/1607.07539."""
  mask = features['mask']
  weighted_mask = features['weighted_mask']
  loss_weights = mask - weighted_mask
  inverse_mask = 1 - mask

  contextual_loss = tf.reduce_sum(
      tf.keras.backend.batch_flatten(
          tf.abs(tf.multiply(loss_weights, gan_model.generated_data) -
                 tf.multiply(loss_weights, labels))), 1)

  perceptual_loss = tfgan.losses.wasserstein_generator_loss(
      gan_model, add_summaries=add_summaries)
  inpaint_loss = contextual_loss + perceptual_loss

  inpaint_loss = tf.reduce_mean(inpaint_loss)
  infilled_image = (tf.multiply(mask, gan_model.real_data) +
                    tf.multiply(inverse_mask, gan_model.generated_data))

  if add_summaries:
    contextual_loss = tf.reduce_mean(contextual_loss)
    perceptual_loss = tf.reduce_mean(perceptual_loss)
    tf.summary.scalar('contextual_loss', contextual_loss)
    tf.summary.scalar('perceptual_loss', perceptual_loss)
    tf.summary.image('mask_image', tf.multiply(mask, gan_model.real_data))
    tf.summary.image('gen_image', gan_model.generated_data)
    tf.summary.image('loss_mask', tf.expand_dims(loss_weights, axis=0))

  tf.summary.image('infill_image', infilled_image)
  return inpaint_loss


def main(_):
  imshape = (FLAGS.resize_h, FLAGS.resize_w)

  params = {
      'batch_size': FLAGS.batch_size,
      'z_shape': [FLAGS.z_dim],
      'learning_rate': FLAGS.learning_rate,
      'input_clip': 1.5,
      'add_summaries': True
  }

  # Make sure we do not save checkpoints while also allowing for distributed
  # training by setting save_checkpoints_steps to one more than the number of
  # total training steps.
  #
  # Note that the RunConfig model_dir should NOT have any checkpoints in that
  # directory. Even though WarmStartSettings are being used under the hood,
  # tensorflow will ignore the WarmStartSettings and try to reload from the
  # model_dir anyway if model_dir contains any checkpoints.
  conf = tf.estimator.RunConfig(model_dir=FLAGS.save_dir + '/train_input',
                                tf_random_seed=FLAGS.seed,
                                save_summary_steps=FLAGS.mod_sum,
                                save_checkpoints_steps=FLAGS.num_steps + 1,
                                save_checkpoints_secs=None,
                                log_step_count_steps=FLAGS.mod_sum)

  estimator = tfgan.estimator.get_latent_gan_estimator(
      faces_gan.generator, faces_gan.discriminator, inpainter_loss,
      tf.train.AdamOptimizer, params, conf, FLAGS.save_dir,
      warmstart_options=FLAGS.warmstart)

  input_fn = get_input_fn(
      FLAGS.data_dir, FLAGS.batch_size, imshape, FLAGS.mask_side_len)
  train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn, max_steps=FLAGS.num_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  tf.app.run(main)
