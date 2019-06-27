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

"""Tests for TF-GAN tfgan.eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_gan as tfgan


def generator_model(inputs):
  return tf.compat.v1.get_variable('dummy_g', initializer=2.0) * inputs


def discriminator_model(inputs, _):
  return tf.compat.v1.get_variable('dummy_d', initializer=2.0) * inputs


def stargan_generator_model(inputs, _):
  return generator_model(inputs)


def get_gan_model():
  # TODO(joelshor): Find a better way of creating a variable scope.
  with tf.compat.v1.variable_scope('generator') as gen_scope:
    pass
  with tf.compat.v1.variable_scope('discriminator') as dis_scope:
    pass
  return tfgan.GANModel(
      generator_inputs=tf.zeros([4, 32, 32, 3]),
      generated_data=tf.zeros([4, 32, 32, 3]),
      generator_variables=[tf.Variable(0), tf.Variable(1)],
      generator_scope=gen_scope,
      generator_fn=generator_model,
      real_data=tf.ones([4, 32, 32, 3]),
      discriminator_real_outputs=tf.ones([1, 2, 3]),
      discriminator_gen_outputs=tf.ones([1, 2, 3]),
      discriminator_variables=[tf.Variable(0)],
      discriminator_scope=dis_scope,
      discriminator_fn=discriminator_model)


def get_stargan_model():
  """Similar to get_gan_model()."""
  # TODO(joelshor): Find a better way of creating a variable scope.
  with tf.compat.v1.variable_scope('discriminator') as dis_scope:
    pass
  with tf.compat.v1.variable_scope('generator') as gen_scope:
    return tfgan.StarGANModel(
        input_data=tf.ones([1, 2, 2, 3]),
        input_data_domain_label=tf.ones([1, 2]),
        generated_data=stargan_generator_model(
            tf.ones([1, 2, 2, 3]), None),
        generated_data_domain_target=tf.ones([1, 2]),
        reconstructed_data=tf.ones([1, 2, 2, 3]),
        discriminator_input_data_source_predication=tf.ones([1]),
        discriminator_generated_data_source_predication=tf.ones([1]),
        discriminator_input_data_domain_predication=tf.ones([1, 2]),
        discriminator_generated_data_domain_predication=tf.ones([1, 2]),
        generator_variables=None,
        generator_scope=gen_scope,
        generator_fn=stargan_generator_model,
        discriminator_variables=None,
        discriminator_scope=dis_scope,
        discriminator_fn=discriminator_model)


def get_cyclegan_model():
  with tf.compat.v1.variable_scope('x2y'):
    model_x2y = get_gan_model()
  with tf.compat.v1.variable_scope('y2x'):
    model_y2x = get_gan_model()
  return tfgan.CycleGANModel(
      model_x2y=model_x2y,
      model_y2x=model_y2x,
      reconstructed_x=tf.zeros([4, 32, 32, 3]),
      reconstructed_y=tf.zeros([4, 32, 32, 3]))


class SummariesTest(tf.test.TestCase):

  def _test_add_gan_model_image_summaries_impl(
      self, get_model_fn, expected_num_summary_ops, model_summaries):
    if tf.executing_eagerly():
      return
    tfgan.eval.add_gan_model_image_summaries(
        get_model_fn(), grid_size=2, model_summaries=model_summaries)
    self.assertEqual(
        expected_num_summary_ops,
        len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.summary.merge_all())

  def test_add_gan_model_image_summaries(self):
    self._test_add_gan_model_image_summaries_impl(get_gan_model, 5, True)

  def test_add_gan_model_image_summaries_no_model(self):
    self._test_add_gan_model_image_summaries_impl(get_gan_model, 2, False)

  def test_cyclegan_image_summaries_dont_work(self):
    with self.assertRaises(ValueError):
      tfgan.eval.add_gan_model_image_summaries(get_cyclegan_model())

  def _test_add_gan_model_summaries_impl(self, get_model_fn,
                                         expected_num_summary_ops):
    if tf.executing_eagerly():
      return
    tfgan.eval.add_gan_model_summaries(get_model_fn())
    self.assertEqual(
        expected_num_summary_ops,
        len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.summary.merge_all())

  def test_add_gan_model_summaries(self):
    self._test_add_gan_model_summaries_impl(get_gan_model, 3)

  def test_add_gan_model_summaries_for_cyclegan(self):
    self._test_add_gan_model_summaries_impl(get_cyclegan_model, 6)

  def _test_add_regularization_loss_summaries_impl(self, get_model_fn,
                                                   expected_num_summary_ops):
    if tf.executing_eagerly():
      return
    tfgan.eval.add_regularization_loss_summaries(get_model_fn())
    self.assertEqual(
        expected_num_summary_ops,
        len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.summary.merge_all())

  def test_add_regularization_loss_summaries(self):
    self._test_add_regularization_loss_summaries_impl(get_gan_model, 2)

  def test_add_regularization_loss_summaries_for_cyclegan(self):
    self._test_add_regularization_loss_summaries_impl(get_cyclegan_model, 4)

  # TODO(joelshor): Add correctness test.
  def _test_add_image_comparison_summaries_impl(self, get_model_fn,
                                                expected_num_summary_ops):
    if tf.executing_eagerly():
      return
    tfgan.eval.add_image_comparison_summaries(
        get_model_fn(), display_diffs=True)

    self.assertEqual(
        expected_num_summary_ops,
        len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.summary.merge_all())

  def test_add_image_comparison_summaries(self):
    self._test_add_image_comparison_summaries_impl(get_gan_model, 1)

  def test_add_image_comparison_summaries_for_cyclegan(self):
    if tf.executing_eagerly():
      return
    tfgan.eval.add_cyclegan_image_summaries(get_cyclegan_model())

    self.assertEqual(
        2, len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.summary.merge_all())

  def test_add_image_comparison_summaries_for_stargan(self):
    if tf.executing_eagerly():
      return
    tfgan.eval.add_stargan_image_summaries(get_stargan_model())

    self.assertEqual(
        1, len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)))

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.summary.merge_all())


if __name__ == '__main__':
  tf.test.main()
