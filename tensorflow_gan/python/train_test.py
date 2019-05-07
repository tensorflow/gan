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

"""Tests for tfgan.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.python.contrib_utils import get_trainable_variables

# Private functions to be tested.
from tensorflow_gan.python.train import generate_stargan_random_domain_target
from tensorflow_gan.python.train import tensor_pool_adjusted_model
import tensorflow_probability as tfp


def generator_model(inputs):
  return tf.compat.v1.get_variable('dummy_g', initializer=2.0) * inputs


class Generator(object):

  def __call__(self, inputs):
    return generator_model(inputs)


def infogan_generator_model(inputs):
  return tf.compat.v1.get_variable('dummy_g', initializer=2.0) * inputs[0]


class InfoGANGenerator(object):

  def __call__(self, inputs):
    return infogan_generator_model(inputs)


def discriminator_model(inputs, _):
  return tf.compat.v1.get_variable('dummy_d', initializer=2.0) * inputs


class Discriminator(object):

  def __call__(self, inputs, _):
    return discriminator_model(inputs, _)


def infogan_discriminator_model(inputs, _):
  return (tf.compat.v1.get_variable(
      'dummy_d', initializer=2.0) * inputs,
          [tfp.distributions.Categorical([1.0])])


class InfoGANDiscriminator(object):

  def __call__(self, inputs, _):
    return infogan_discriminator_model(inputs, _)


def acgan_discriminator_model(inputs, _, num_classes=10):
  return (
      discriminator_model(inputs, _),
      tf.one_hot(
          # TODO(haeusser): infer batch size from input
          tf.random.uniform([3], maxval=num_classes, dtype=tf.int32),
          num_classes))


class ACGANDiscriminator(object):

  def __call__(self, inputs, _, num_classes=10):
    return (
        discriminator_model(inputs, _),
        tf.one_hot(
            # TODO(haeusser): infer batch size from input
            tf.random.uniform([3], maxval=num_classes, dtype=tf.int32),
            num_classes))


def stargan_generator_model(inputs, _):
  """Dummy generator for StarGAN."""

  return tf.compat.v1.get_variable('dummy_g', initializer=0.5) * inputs


class StarGANGenerator(object):

  def __call__(self, inputs, _):
    return stargan_generator_model(inputs, _)


def stargan_discriminator_model(inputs, num_domains):
  """Differentiable dummy discriminator for StarGAN."""
  hidden = tf.compat.v1.layers.flatten(inputs)
  output_src = tf.reduce_mean(input_tensor=hidden, axis=1)
  output_cls = tf.compat.v1.layers.dense(hidden, num_domains)
  return output_src, output_cls


class StarGANDiscriminator(object):

  def __call__(self, inputs, num_domains):
    return stargan_discriminator_model(inputs, num_domains)


def get_gan_model():
  # TODO(joelshor): Find a better way of creating a variable scope.
  with tf.compat.v1.variable_scope('generator') as gen_scope:
    pass
  with tf.compat.v1.variable_scope('discriminator') as dis_scope:
    pass
  return tfgan.GANModel(
      generator_inputs=None,
      generated_data=None,
      generator_variables=None,
      generator_scope=gen_scope,
      generator_fn=generator_model,
      real_data=tf.ones([1, 2, 3]),
      discriminator_real_outputs=tf.ones([1, 2, 3]),
      discriminator_gen_outputs=tf.ones([1, 2, 3]),
      discriminator_variables=None,
      discriminator_scope=dis_scope,
      discriminator_fn=discriminator_model)


def get_callable_gan_model():
  ganmodel = get_gan_model()
  return ganmodel._replace(
      generator_fn=Generator(), discriminator_fn=Discriminator())


def create_gan_model():
  return tfgan.gan_model(
      generator_model,
      discriminator_model,
      real_data=tf.zeros([1, 2]),
      generator_inputs=tf.random.normal([1, 2]))


def create_callable_gan_model():
  return tfgan.gan_model(
      Generator(),
      Discriminator(),
      real_data=tf.zeros([1, 2]),
      generator_inputs=tf.random.normal([1, 2]))


def get_infogan_model():
  return tfgan.InfoGANModel(
      *get_gan_model(),
      structured_generator_inputs=[tf.constant(0)],
      predicted_distributions=[tfp.distributions.Categorical([1.0])],
      discriminator_and_aux_fn=infogan_discriminator_model)


def get_callable_infogan_model():
  return tfgan.InfoGANModel(
      *get_callable_gan_model(),
      structured_generator_inputs=[tf.constant(0)],
      predicted_distributions=[tfp.distributions.Categorical([1.0])],
      discriminator_and_aux_fn=infogan_discriminator_model)


def create_infogan_model():
  return tfgan.infogan_model(
      infogan_generator_model,
      infogan_discriminator_model,
      real_data=tf.zeros([1, 2]),
      unstructured_generator_inputs=[],
      structured_generator_inputs=[tf.random.normal([1, 2])])


def create_callable_infogan_model():
  return tfgan.infogan_model(
      InfoGANGenerator(),
      InfoGANDiscriminator(),
      real_data=tf.zeros([1, 2]),
      unstructured_generator_inputs=[],
      structured_generator_inputs=[tf.random.normal([1, 2])])


def get_acgan_model():
  return tfgan.ACGANModel(
      *get_gan_model(),
      one_hot_labels=tf.one_hot([0, 1, 2], 10),
      discriminator_real_classification_logits=tf.one_hot([0, 1, 3], 10),
      discriminator_gen_classification_logits=tf.one_hot([0, 1, 4], 10))


def get_callable_acgan_model():
  return tfgan.ACGANModel(
      *get_callable_gan_model(),
      one_hot_labels=tf.one_hot([0, 1, 2], 10),
      discriminator_real_classification_logits=tf.one_hot([0, 1, 3], 10),
      discriminator_gen_classification_logits=tf.one_hot([0, 1, 4], 10))


def create_acgan_model():
  return tfgan.acgan_model(
      generator_model,
      acgan_discriminator_model,
      real_data=tf.zeros([1, 2]),
      generator_inputs=tf.random.normal([1, 2]),
      one_hot_labels=tf.one_hot([0, 1, 2], 10))


def create_callable_acgan_model():
  return tfgan.acgan_model(
      Generator(),
      ACGANDiscriminator(),
      real_data=tf.zeros([1, 2]),
      generator_inputs=tf.random.normal([1, 2]),
      one_hot_labels=tf.one_hot([0, 1, 2], 10))


def get_cyclegan_model():
  return tfgan.CycleGANModel(
      model_x2y=get_gan_model(),
      model_y2x=get_gan_model(),
      reconstructed_x=tf.ones([1, 2, 3]),
      reconstructed_y=tf.zeros([1, 2, 3]))


def get_callable_cyclegan_model():
  return tfgan.CycleGANModel(
      model_x2y=get_callable_gan_model(),
      model_y2x=get_callable_gan_model(),
      reconstructed_x=tf.ones([1, 2, 3]),
      reconstructed_y=tf.zeros([1, 2, 3]))


def create_cyclegan_model():
  return tfgan.cyclegan_model(
      generator_model,
      discriminator_model,
      data_x=tf.zeros([1, 2]),
      data_y=tf.ones([1, 2]))


def create_callable_cyclegan_model():
  return tfgan.cyclegan_model(
      Generator(),
      Discriminator(),
      data_x=tf.zeros([1, 2]),
      data_y=tf.ones([1, 2]))


def get_stargan_model():
  """Similar to get_gan_model()."""
  # TODO(joelshor): Find a better way of creating a variable scope.
  with tf.compat.v1.variable_scope('generator') as gen_scope:
    pass
  with tf.compat.v1.variable_scope('discriminator') as dis_scope:
    pass
  return tfgan.StarGANModel(
      input_data=tf.ones([1, 2, 2, 3]),
      input_data_domain_label=tf.ones([1, 2]),
      generated_data=tf.ones([1, 2, 2, 3]),
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
      discriminator_fn=stargan_discriminator_model)


def get_callable_stargan_model():
  model = get_stargan_model()
  return model._replace(
      generator_fn=StarGANGenerator(), discriminator_fn=StarGANDiscriminator())


def create_stargan_model():
  return tfgan.stargan_model(stargan_generator_model,
                             stargan_discriminator_model, tf.ones([1, 2, 2, 3]),
                             tf.ones([1, 2]))


def create_callable_stargan_model():
  return tfgan.stargan_model(StarGANGenerator(), StarGANDiscriminator(),
                             tf.ones([1, 2, 2, 3]), tf.ones([1, 2]))


def get_sync_optimizer():
  return tf.compat.v1.train.SyncReplicasOptimizer(
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0),
      replicas_to_aggregate=1)


def get_sync_optimizer_hook_type():
  dummy_opt = get_sync_optimizer()
  dummy_hook = dummy_opt.make_session_run_hook(is_chief=True)
  return type(dummy_hook)


class GANModelTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `gan_model`."""

  @parameterized.named_parameters(
      ('gan', get_gan_model, tfgan.GANModel),
      ('callable_gan', get_callable_gan_model, tfgan.GANModel),
      ('infogan', get_infogan_model, tfgan.InfoGANModel),
      ('callable_infogan', get_callable_infogan_model, tfgan.InfoGANModel),
      ('acgan', get_acgan_model, tfgan.ACGANModel),
      ('callable_acgan', get_callable_acgan_model, tfgan.ACGANModel),
      ('cyclegan', get_cyclegan_model, tfgan.CycleGANModel),
      ('callable_cyclegan', get_callable_cyclegan_model, tfgan.CycleGANModel),
      ('stargan', get_stargan_model, tfgan.StarGANModel),
      ('callabel_stargan', get_callable_stargan_model, tfgan.StarGANModel))
  def test_output_type(self, create_fn, expected_tuple_type):
    """Test that output type is as expected."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    self.assertIsInstance(create_fn(), expected_tuple_type)

  def test_no_shape_check(self):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    def dummy_generator_model(_):
      return (None, None)

    def dummy_discriminator_model(data, conditioning):  # pylint: disable=unused-argument
      return 1

    with self.assertRaisesRegexp(AttributeError, 'object has no attribute'):
      tfgan.gan_model(
          dummy_generator_model,
          dummy_discriminator_model,
          real_data=tf.zeros([1, 2]),
          generator_inputs=tf.zeros([1]),
          check_shapes=True)
    tfgan.gan_model(
        dummy_generator_model,
        dummy_discriminator_model,
        real_data=tf.zeros([1, 2]),
        generator_inputs=tf.zeros([1]),
        check_shapes=False)

  def test_multiple_models(self):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    # Verify that creating 2 GANModels with the same scope names does not create
    # double the variables.
    create_gan_model()
    variables_1 = tf.compat.v1.global_variables()
    create_gan_model()
    variables_2 = tf.compat.v1.global_variables()
    self.assertEqual(variables_1, variables_2)


class StarGANModelTest(tf.test.TestCase):
  """Tests for `stargan_model`."""

  @staticmethod
  def create_input_and_label_tensor(batch_size, img_size, c_size, num_domains):
    input_tensor_list = []
    label_tensor_list = []
    for _ in range(num_domains):
      input_tensor_list.append(
          tf.random.uniform((batch_size, img_size, img_size, c_size)))
      domain_idx = tf.random.uniform([batch_size],
                                     minval=0,
                                     maxval=num_domains,
                                     dtype=tf.int32)
      label_tensor_list.append(tf.one_hot(domain_idx, num_domains))
    return input_tensor_list, label_tensor_list

  def test_generate_stargan_random_domain_target(self):
    batch_size = 8
    domain_numbers = 3

    target_tensor = generate_stargan_random_domain_target(
        batch_size, domain_numbers)

    with self.cached_session() as sess:
      targets = sess.run(target_tensor)
      self.assertTupleEqual((batch_size, domain_numbers), targets.shape)
      for target in targets:
        self.assertEqual(1, np.sum(target))
        self.assertEqual(1, np.max(target))

  def test_stargan_model_output_type(self):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    batch_size = 2
    img_size = 16
    c_size = 3
    num_domains = 5

    input_tensor, label_tensor = StarGANModelTest.create_input_and_label_tensor(
        batch_size, img_size, c_size, num_domains)
    model = tfgan.stargan_model(
        generator_fn=stargan_generator_model,
        discriminator_fn=stargan_discriminator_model,
        input_data=input_tensor,
        input_data_domain_label=label_tensor)

    self.assertIsInstance(model, tfgan.StarGANModel)
    self.assertTrue(isinstance(model.discriminator_variables, list))
    self.assertTrue(isinstance(model.generator_variables, list))
    self.assertIsInstance(model.discriminator_scope, tf.compat.v1.VariableScope)
    self.assertTrue(model.generator_scope, tf.compat.v1.VariableScope)
    self.assertTrue(callable(model.discriminator_fn))
    self.assertTrue(callable(model.generator_fn))

  def test_stargan_model_generator_output(self):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    batch_size = 2
    img_size = 16
    c_size = 3
    num_domains = 5

    input_tensor, label_tensor = StarGANModelTest.create_input_and_label_tensor(
        batch_size, img_size, c_size, num_domains)
    model = tfgan.stargan_model(
        generator_fn=stargan_generator_model,
        discriminator_fn=stargan_discriminator_model,
        input_data=input_tensor,
        input_data_domain_label=label_tensor)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())

      input_data, generated_data, reconstructed_data = sess.run(
          [model.input_data, model.generated_data, model.reconstructed_data])
      self.assertTupleEqual(
          (batch_size * num_domains, img_size, img_size, c_size),
          input_data.shape)
      self.assertTupleEqual(
          (batch_size * num_domains, img_size, img_size, c_size),
          generated_data.shape)
      self.assertTupleEqual(
          (batch_size * num_domains, img_size, img_size, c_size),
          reconstructed_data.shape)

  def test_stargan_model_discriminator_output(self):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    batch_size = 2
    img_size = 16
    c_size = 3
    num_domains = 5

    input_tensor, label_tensor = StarGANModelTest.create_input_and_label_tensor(
        batch_size, img_size, c_size, num_domains)
    model = tfgan.stargan_model(
        generator_fn=stargan_generator_model,
        discriminator_fn=stargan_discriminator_model,
        input_data=input_tensor,
        input_data_domain_label=label_tensor)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())

      disc_input_data_source_pred, disc_gen_data_source_pred = sess.run([
          model.discriminator_input_data_source_predication,
          model.discriminator_generated_data_source_predication
      ])
      self.assertEqual(1, len(disc_input_data_source_pred.shape))
      self.assertEqual(batch_size * num_domains,
                       disc_input_data_source_pred.shape[0])
      self.assertEqual(1, len(disc_gen_data_source_pred.shape))
      self.assertEqual(batch_size * num_domains,
                       disc_gen_data_source_pred.shape[0])

      input_label, disc_input_label, gen_label, disc_gen_label = sess.run([
          model.input_data_domain_label,
          model.discriminator_input_data_domain_predication,
          model.generated_data_domain_target,
          model.discriminator_generated_data_domain_predication
      ])
      self.assertTupleEqual((batch_size * num_domains, num_domains),
                            input_label.shape)
      self.assertTupleEqual((batch_size * num_domains, num_domains),
                            disc_input_label.shape)
      self.assertTupleEqual((batch_size * num_domains, num_domains),
                            gen_label.shape)
      self.assertTupleEqual((batch_size * num_domains, num_domains),
                            disc_gen_label.shape)


class GANLossTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `gan_loss`."""

  @parameterized.named_parameters(
      ('gan', get_gan_model),
      ('callable_gan', get_callable_gan_model),
      ('infogan', get_infogan_model),
      ('callable_infogan', get_callable_infogan_model),
      ('acgan', get_acgan_model),
      ('callable_acgan', get_callable_acgan_model),
  )
  def test_output_type(self, get_gan_model_fn):
    """Test output type."""
    loss = tfgan.gan_loss(get_gan_model_fn(), add_summaries=True)
    self.assertIsInstance(loss, tfgan.GANLoss)
    self.assertEqual(0, loss.generator_loss.shape.ndims)
    self.assertEqual(0, loss.discriminator_loss.shape.ndims)
    if not tf.executing_eagerly():  # Collections don't work in eager.
      self.assertNotEmpty(
          tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

  @parameterized.named_parameters(
      ('cyclegan', create_cyclegan_model),
      ('callable_cyclegan', create_callable_cyclegan_model),
  )
  def test_cyclegan_output_type(self, get_gan_model_fn):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    loss = tfgan.cyclegan_loss(get_gan_model_fn(), add_summaries=True)
    self.assertIsInstance(loss, tfgan.CycleGANLoss)
    self.assertEqual(0, loss.loss_x2y.discriminator_loss.shape.ndims)
    self.assertEqual(0, loss.loss_x2y.generator_loss.shape.ndims)
    self.assertEqual(0, loss.loss_y2x.discriminator_loss.shape.ndims)
    self.assertEqual(0, loss.loss_y2x.generator_loss.shape.ndims)
    if not tf.executing_eagerly():  # Collections don't work in eager.
      self.assertNotEmpty(
          tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

  @parameterized.named_parameters(
      ('infogan', get_infogan_model,
       {'mutual_information_penalty_weight': 1.0}),
      ('acgan', get_acgan_model,
       {'aux_cond_generator_weight': 1.0,
        'aux_cond_discriminator_weight': 1.0}),
  )
  def test_reduction(self, get_gan_model_fn, kwargs):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    loss = tfgan.gan_loss(
        get_gan_model_fn(),
        reduction=tf.compat.v1.losses.Reduction.NONE,
        **kwargs)
    self.assertIsInstance(loss, tfgan.GANLoss)
    self.assertEqual(3, loss.generator_loss.shape.ndims)
    self.assertEqual(3, loss.discriminator_loss.shape.ndims)

  def test_reduction_cyclegan(self):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    loss = tfgan.cyclegan_loss(
        create_cyclegan_model(), reduction=tf.compat.v1.losses.Reduction.NONE)
    self.assertIsInstance(loss, tfgan.CycleGANLoss)
    self.assertEqual(2, loss.loss_x2y.discriminator_loss.shape.ndims)
    self.assertEqual(2, loss.loss_x2y.generator_loss.shape.ndims)
    self.assertEqual(2, loss.loss_y2x.discriminator_loss.shape.ndims)
    self.assertEqual(2, loss.loss_y2x.generator_loss.shape.ndims)

  def test_no_reduction_or_add_summaries_loss(self):
    def loss_fn(_):
      return 0
    tfgan.gan_loss(get_gan_model(), loss_fn, loss_fn)

  def test_args_passed_in_correctly(self):
    def loss_fn(gan_model, add_summaries):
      del gan_model
      self.assertFalse(add_summaries)
      return 0
    tfgan.gan_loss(get_gan_model(), loss_fn, loss_fn, add_summaries=False)

  @parameterized.named_parameters(
      ('gan', create_gan_model, False),
      ('gan_one_sided', create_gan_model, True),
      ('callable_gan', create_callable_gan_model, False),
      ('callable_gan_one_sided', create_callable_gan_model, True),
      ('infogan', create_infogan_model, False),
      ('infogan_one_sided', create_infogan_model, True),
      ('callable_infogan', create_callable_infogan_model, False),
      ('callable_infogan_one_sided', create_callable_infogan_model, True),
      ('acgan', create_acgan_model, False),
      ('acgan_one_sided', create_acgan_model, True),
      ('callable_acgan', create_callable_acgan_model, False),
      ('callable_acgan_one_sided', create_callable_acgan_model, True),
  )
  def test_grad_penalty(self, create_gan_model_fn, one_sided):
    """Test gradient penalty option."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    model = create_gan_model_fn()
    loss = tfgan.gan_loss(model)
    loss_gp = tfgan.gan_loss(
        model,
        gradient_penalty_weight=1.0,
        gradient_penalty_one_sided=one_sided)
    self.assertIsInstance(loss_gp, tfgan.GANLoss)

    # Check values.
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      loss_gen_np, loss_gen_gp_np = sess.run(
          [loss.generator_loss, loss_gp.generator_loss])
      loss_dis_np, loss_dis_gp_np = sess.run(
          [loss.discriminator_loss, loss_gp.discriminator_loss])

    self.assertEqual(loss_gen_np, loss_gen_gp_np)
    self.assertLess(loss_dis_np, loss_dis_gp_np)

  @parameterized.named_parameters(
      ('infogan', get_infogan_model),
      ('callable_infogan', get_callable_infogan_model),
  )
  def test_mutual_info_penalty(self, create_gan_model_fn):
    """Test mutual information penalty option."""
    tfgan.gan_loss(
        create_gan_model_fn(),
        mutual_information_penalty_weight=tf.constant(1.0))

  @parameterized.named_parameters(
      ('gan', get_gan_model),
      ('callable_gan', get_callable_gan_model),
      ('infogan', get_infogan_model),
      ('callable_infogan', get_callable_infogan_model),
      ('acgan', get_acgan_model),
      ('callable_acgan', get_callable_acgan_model),
  )
  def test_regularization_helper(self, get_gan_model_fn):
    """Test regularization loss."""
    if tf.executing_eagerly():
      # Regularization capture requires collections.
      return
    # Evaluate losses without regularization.
    no_reg_loss = tfgan.gan_loss(get_gan_model_fn())
    with self.cached_session() as sess:
      no_reg_loss_g, no_reg_loss_d = sess.run([no_reg_loss.generator_loss,
                                               no_reg_loss.discriminator_loss])

    with tf.compat.v1.name_scope(get_gan_model_fn().generator_scope.name):
      tf.compat.v1.add_to_collection(
          tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, tf.constant(3.0))
    with tf.compat.v1.name_scope(get_gan_model_fn().discriminator_scope.name):
      tf.compat.v1.add_to_collection(
          tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, tf.constant(2.0))

    # Check that losses now include the correct regularization values.
    reg_loss = tfgan.gan_loss(get_gan_model_fn())
    with self.cached_session() as sess:
      reg_loss_g, reg_loss_d = sess.run([reg_loss.generator_loss,
                                         reg_loss.discriminator_loss])

    self.assertEqual(3.0, reg_loss_g - no_reg_loss_g)
    self.assertEqual(2.0, reg_loss_d - no_reg_loss_d)

  @parameterized.named_parameters(
      ('notcallable', create_acgan_model),
      ('callable', create_callable_acgan_model),
  )
  def test_acgan(self, create_gan_model_fn):
    """Test that ACGAN models work."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    model = create_gan_model_fn()
    loss = tfgan.gan_loss(model)
    loss_ac_gen = tfgan.gan_loss(model, aux_cond_generator_weight=1.0)
    loss_ac_dis = tfgan.gan_loss(model, aux_cond_discriminator_weight=1.0)
    self.assertIsInstance(loss, tfgan.GANLoss)
    self.assertIsInstance(loss_ac_gen, tfgan.GANLoss)
    self.assertIsInstance(loss_ac_dis, tfgan.GANLoss)

    # Check values.
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      loss_gen_np, loss_ac_gen_gen_np, loss_ac_dis_gen_np = sess.run([
          loss.generator_loss, loss_ac_gen.generator_loss,
          loss_ac_dis.generator_loss
      ])
      loss_dis_np, loss_ac_gen_dis_np, loss_ac_dis_dis_np = sess.run([
          loss.discriminator_loss, loss_ac_gen.discriminator_loss,
          loss_ac_dis.discriminator_loss
      ])

    self.assertLess(loss_gen_np, loss_dis_np)
    self.assertTrue(np.isscalar(loss_ac_gen_gen_np))
    self.assertTrue(np.isscalar(loss_ac_dis_gen_np))
    self.assertTrue(np.isscalar(loss_ac_gen_dis_np))
    self.assertTrue(np.isscalar(loss_ac_dis_dis_np))

  @parameterized.named_parameters(
      ('notcallable', create_cyclegan_model),
      ('callable', create_callable_cyclegan_model),
  )
  def test_cyclegan(self, create_gan_model_fn):
    """Test that CycleGan models work."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    model = create_gan_model_fn()
    loss = tfgan.cyclegan_loss(model)
    self.assertIsInstance(loss, tfgan.CycleGANLoss)

    # Check values.
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      (loss_x2y_gen_np, loss_x2y_dis_np, loss_y2x_gen_np,
       loss_y2x_dis_np) = sess.run([
           loss.loss_x2y.generator_loss, loss.loss_x2y.discriminator_loss,
           loss.loss_y2x.generator_loss, loss.loss_y2x.discriminator_loss
       ])

    self.assertGreater(loss_x2y_gen_np, loss_x2y_dis_np)
    self.assertGreater(loss_y2x_gen_np, loss_y2x_dis_np)
    self.assertTrue(np.isscalar(loss_x2y_gen_np))
    self.assertTrue(np.isscalar(loss_x2y_dis_np))
    self.assertTrue(np.isscalar(loss_y2x_gen_np))
    self.assertTrue(np.isscalar(loss_y2x_dis_np))

  @parameterized.named_parameters(
      ('notcallable', create_stargan_model),
      ('callable', create_callable_stargan_model),
  )
  def test_stargan(self, create_gan_model_fn):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    model = create_gan_model_fn()
    model_loss = tfgan.stargan_loss(model)

    self.assertIsInstance(model_loss, tfgan.GANLoss)

    with self.cached_session() as sess:

      sess.run(tf.compat.v1.global_variables_initializer())

      gen_loss, disc_loss = sess.run(
          [model_loss.generator_loss, model_loss.discriminator_loss])

      self.assertTrue(np.isscalar(gen_loss))
      self.assertTrue(np.isscalar(disc_loss))

  @parameterized.named_parameters(
      ('gan', create_gan_model),
      ('callable_gan', create_callable_gan_model),
      ('infogan', create_infogan_model),
      ('callable_infogan', create_callable_infogan_model),
      ('acgan', create_acgan_model),
      ('callable_acgan', create_callable_acgan_model),
  )
  def test_tensor_pool(self, create_gan_model_fn):
    """Test tensor pool option."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    model = create_gan_model_fn()
    tensor_pool_fn = lambda x: tfgan.features.tensor_pool(x, pool_size=5)
    loss = tfgan.gan_loss(model, tensor_pool_fn=tensor_pool_fn)
    self.assertIsInstance(loss, tfgan.GANLoss)

    # Check values.
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      for _ in range(10):
        sess.run([loss.generator_loss, loss.discriminator_loss])

  def test_discriminator_only_sees_pool(self):
    """Checks that discriminator only sees pooled values."""
    if tf.executing_eagerly():
      # This test involves '.op', which doesn't work in eager.
      return

    def checker_gen_fn(_):
      return tf.constant(0.0)

    model = tfgan.gan_model(
        checker_gen_fn,
        discriminator_model,
        real_data=tf.zeros([]),
        generator_inputs=tf.random.normal([]))

    def tensor_pool_fn(_):
      return (tf.random.uniform([]), tf.random.uniform([]))

    def checker_dis_fn(inputs, _):
      """Discriminator that checks that it only sees pooled Tensors."""

      def _is_constant(tensor):
        """Returns `True` if the Tensor is a constant."""
        return tensor.op.type == 'Const'

      self.assertFalse(_is_constant(inputs))
      return inputs

    model = model._replace(discriminator_fn=checker_dis_fn)
    tfgan.gan_loss(model, tensor_pool_fn=tensor_pool_fn)

  def test_doesnt_crash_when_in_nested_scope(self):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    with tf.compat.v1.variable_scope('outer_scope'):
      gan_model = tfgan.gan_model(
          generator_model,
          discriminator_model,
          real_data=tf.zeros([1, 2]),
          generator_inputs=tf.random.normal([1, 2]))

      # This should work inside a scope.
      tfgan.gan_loss(gan_model, gradient_penalty_weight=1.0)

    # This should also work outside a scope.
    tfgan.gan_loss(gan_model, gradient_penalty_weight=1.0)


class TensorPoolAdjusteModelTest(tf.test.TestCase):

  def _check_tensor_pool_adjusted_model_outputs(self, tensor1, tensor2,
                                                pool_size):
    history_values = []
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      for i in range(2 * pool_size):
        t1, t2 = sess.run([tensor1, tensor2])
        history_values.append(t1)
        if i < pool_size:
          # For [0, pool_size), the pool is not full, tensor1 should be equal
          # to tensor2 as the pool.
          self.assertAllEqual(t1, t2)
        else:
          # For [pool_size, ?), the pool is full, tensor2 must be equal to some
          # historical values of tensor1 (which is previously stored in the
          # pool).
          self.assertTrue(any((v == t2).all() for v in history_values))

  def _make_new_model_and_check(self, model, pool_size):
    pool_fn = lambda x: tfgan.features.tensor_pool(x, pool_size=pool_size)
    new_model = tensor_pool_adjusted_model(model, pool_fn)
    # 'Generator/dummy_g:0' and 'Discriminator/dummy_d:0'
    if not tf.executing_eagerly():  # Collections don't work in eager.
      self.assertEqual(
          2, len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.VARIABLES)))
    self.assertIsNot(new_model.discriminator_gen_outputs,
                     model.discriminator_gen_outputs)

    return new_model

  def test_tensor_pool_adjusted_model_gan(self):
    """Test `_tensor_pool_adjusted_model` for gan model."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    pool_size = 5
    model = create_gan_model()
    new_model = self._make_new_model_and_check(model, pool_size)

    # Check values.
    self._check_tensor_pool_adjusted_model_outputs(
        model.discriminator_gen_outputs, new_model.discriminator_gen_outputs,
        pool_size)

  def test_tensor_pool_adjusted_model_infogan(self):
    """Test _tensor_pool_adjusted_model for infogan model."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    pool_size = 5
    model = create_infogan_model()
    new_model = self._make_new_model_and_check(model, pool_size)

    # Check values.
    self.assertIsNot(new_model.predicted_distributions,
                     model.predicted_distributions)
    self._check_tensor_pool_adjusted_model_outputs(
        model.discriminator_gen_outputs, new_model.discriminator_gen_outputs,
        pool_size)

  def test_tensor_pool_adjusted_model_acgan(self):
    """Test _tensor_pool_adjusted_model for acgan model."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    pool_size = 5
    model = create_acgan_model()
    new_model = self._make_new_model_and_check(model, pool_size)

    # Check values.
    self.assertIsNot(new_model.discriminator_gen_classification_logits,
                     model.discriminator_gen_classification_logits)
    self._check_tensor_pool_adjusted_model_outputs(
        model.discriminator_gen_outputs, new_model.discriminator_gen_outputs,
        pool_size)


class GANTrainOpsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `gan_train_ops`."""

  @parameterized.named_parameters(
      ('gan', create_gan_model),
      ('callable_gan', create_callable_gan_model),
      ('infogan', create_infogan_model),
      ('callable_infogan', create_callable_infogan_model),
      ('acgan', create_acgan_model),
      ('callable_acgan', create_callable_acgan_model),
  )
  def test_output_type(self, create_gan_model_fn):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    model = create_gan_model_fn()
    loss = tfgan.gan_loss(model)

    g_opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    d_opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    train_ops = tfgan.gan_train_ops(
        model,
        loss,
        g_opt,
        d_opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True)

    self.assertIsInstance(train_ops, tfgan.GANTrainOps)

    # Make sure there are no training hooks populated accidentally.
    self.assertEmpty(train_ops.train_hooks)

  # TODO(joelshor): Add a test to check that custom update op is run.
  @parameterized.named_parameters(
      ('gan', create_gan_model, False),
      ('gan_provideupdates', create_gan_model, True),
      ('callable_gan', create_callable_gan_model, False),
      ('callable_gan_provideupdates', create_callable_gan_model, True),
      ('infogan', create_infogan_model, False),
      ('infogan_provideupdates', create_infogan_model, True),
      ('callable_infogan', create_callable_infogan_model, False),
      ('callable_infogan_provideupdates', create_callable_infogan_model, True),
      ('acgan', create_acgan_model, False),
      ('acgan_provideupdates', create_acgan_model, True),
      ('callable_acgan', create_callable_acgan_model, False),
      ('callable_acgan_provideupdates', create_callable_acgan_model, True),
  )
  def test_unused_update_ops(self, create_gan_model_fn, provide_update_ops):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    model = create_gan_model_fn()
    loss = tfgan.gan_loss(model)

    # Add generator and discriminator update tf.
    with tf.compat.v1.variable_scope(model.generator_scope):
      gen_update_count = tf.compat.v1.get_variable('gen_count', initializer=0)
      gen_update_op = gen_update_count.assign_add(1)
      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS,
                                     gen_update_op)
    with tf.compat.v1.variable_scope(model.discriminator_scope):
      dis_update_count = tf.compat.v1.get_variable('dis_count', initializer=0)
      dis_update_op = dis_update_count.assign_add(1)
      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS,
                                     dis_update_op)

    # Add an update op outside the generator and discriminator scopes.
    if provide_update_ops:
      kwargs = {'update_ops': [tf.constant(1.0), gen_update_op, dis_update_op]}
    else:
      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS,
                                     tf.constant(1.0))
      kwargs = {}

    g_opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    d_opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)

    with self.assertRaisesRegexp(ValueError, 'There are unused update ops:'):
      tfgan.gan_train_ops(
          model, loss, g_opt, d_opt, check_for_unused_update_ops=True, **kwargs)
    train_ops = tfgan.gan_train_ops(
        model, loss, g_opt, d_opt, check_for_unused_update_ops=False, **kwargs)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertEqual(0, sess.run(gen_update_count))
      self.assertEqual(0, sess.run(dis_update_count))

      sess.run(train_ops.generator_train_op)
      self.assertEqual(1, sess.run(gen_update_count))
      self.assertEqual(0, sess.run(dis_update_count))

      sess.run(train_ops.discriminator_train_op)
      self.assertEqual(1, sess.run(gen_update_count))
      self.assertEqual(1, sess.run(dis_update_count))

  @parameterized.named_parameters(
      ('gan', create_gan_model, False),
      ('callable_gan', create_callable_gan_model, False),
      ('infogan', create_infogan_model, False),
      ('callable_infogan', create_callable_infogan_model, False),
      ('acgan', create_acgan_model, False),
      ('callable_acgan', create_callable_acgan_model, False),
      ('gan_canbeint32', create_gan_model, True),
  )
  def test_sync_replicas(self, create_gan_model_fn, create_global_step):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    model = create_gan_model_fn()
    loss = tfgan.gan_loss(model)
    num_trainable_vars = len(get_trainable_variables())

    if create_global_step:
      gstep = tf.compat.v1.get_variable(
          'custom_gstep',
          dtype=tf.int32,
          initializer=0,
          trainable=False)
      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.GLOBAL_STEP, gstep)

    g_opt = get_sync_optimizer()
    d_opt = get_sync_optimizer()
    train_ops = tfgan.gan_train_ops(
        model, loss, generator_optimizer=g_opt, discriminator_optimizer=d_opt)
    self.assertIsInstance(train_ops, tfgan.GANTrainOps)
    # No new trainable variables should have been added.
    self.assertLen(get_trainable_variables(), num_trainable_vars)

    # Sync hooks should be populated in the GANTraintf.
    self.assertLen(train_ops.train_hooks, 2)
    for hook in train_ops.train_hooks:
      self.assertIsInstance(hook, get_sync_optimizer_hook_type())
    sync_opts = [hook._sync_optimizer for hook in train_ops.train_hooks]
    self.assertSetEqual(frozenset(sync_opts), frozenset((g_opt, d_opt)))

    g_sync_init_op = g_opt.get_init_tokens_op(num_tokens=1)
    d_sync_init_op = d_opt.get_init_tokens_op(num_tokens=1)

    # Check that update op is run properly.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())

      sess.run(g_opt.chief_init_op)
      sess.run(d_opt.chief_init_op)

      gstep_before = sess.run(global_step)

      # Start required queue runner for SyncReplicasOptimizer.
      coord = tf.train.Coordinator()
      g_threads = g_opt.get_chief_queue_runner().create_threads(sess, coord)
      d_threads = d_opt.get_chief_queue_runner().create_threads(sess, coord)

      sess.run(g_sync_init_op)
      sess.run(d_sync_init_op)

      sess.run(train_ops.generator_train_op)
      # Check that global step wasn't incremented.
      self.assertEqual(gstep_before, sess.run(global_step))

      sess.run(train_ops.discriminator_train_op)
      # Check that global step wasn't incremented.
      self.assertEqual(gstep_before, sess.run(global_step))

      coord.request_stop()
      coord.join(g_threads + d_threads)

  @parameterized.named_parameters(
      ('is_chief', True),
      ('is_not_chief', False),
  )
  def test_is_chief_in_train_hooks(self, is_chief):
    """Make sure is_chief is propagated correctly to sync hooks."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    model = create_gan_model()
    loss = tfgan.gan_loss(model)
    g_opt = get_sync_optimizer()
    d_opt = get_sync_optimizer()
    train_ops = tfgan.gan_train_ops(
        model,
        loss,
        g_opt,
        d_opt,
        is_chief=is_chief,
        summarize_gradients=True,
        colocate_gradients_with_ops=True)

    self.assertLen(train_ops.train_hooks, 2)

    for hook in train_ops.train_hooks:
      self.assertIsInstance(hook, get_sync_optimizer_hook_type())
    is_chief_list = [hook._is_chief for hook in train_ops.train_hooks]
    self.assertListEqual(is_chief_list, [is_chief, is_chief])


class GANTrainTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for `gan_train`."""

  def _gan_train_ops(self, generator_add, discriminator_add):
    step = tf.compat.v1.train.create_global_step()
    # Increment the global count every time a train op is run so we can count
    # the number of times they're run.
    # NOTE: `use_locking=True` is required to avoid race conditions with
    # joint training.
    train_ops = tfgan.GANTrainOps(
        generator_train_op=step.assign_add(generator_add, use_locking=True),
        discriminator_train_op=step.assign_add(
            discriminator_add, use_locking=True),
        global_step_inc_op=step.assign_add(1))
    return train_ops

  @parameterized.named_parameters(
      ('gan', create_gan_model),
      ('callable_gan', create_callable_gan_model),
      ('infogan', create_infogan_model),
      ('callable_infogan', create_callable_infogan_model),
      ('acgan', create_acgan_model),
      ('callable_acgan', create_callable_acgan_model),
  )
  def test_run_helper(self, create_gan_model_fn):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    tf.compat.v1.random.set_random_seed(1234)
    model = create_gan_model_fn()
    loss = tfgan.gan_loss(model)

    g_opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    d_opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    train_ops = tfgan.gan_train_ops(model, loss, g_opt, d_opt)

    final_step = tfgan.gan_train(
        train_ops, logdir='', hooks=[tf.estimator.StopAtStepHook(num_steps=2)])
    self.assertTrue(np.isscalar(final_step))
    self.assertEqual(2, final_step)

  @parameterized.named_parameters(
      ('seq_train_steps', tfgan.get_sequential_train_hooks),
      ('efficient_seq_train_steps', tfgan.get_joint_train_hooks),
  )
  def test_multiple_steps(self, get_hooks_fn_fn):
    """Test multiple train steps."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    train_ops = self._gan_train_ops(generator_add=10, discriminator_add=100)
    train_steps = tfgan.GANTrainSteps(
        generator_train_steps=3, discriminator_train_steps=4)
    final_step = tfgan.gan_train(
        train_ops,
        get_hooks_fn=get_hooks_fn_fn(train_steps),
        logdir='',
        hooks=[tf.estimator.StopAtStepHook(num_steps=1)])

    self.assertTrue(np.isscalar(final_step))
    self.assertEqual(1 + 3 * 10 + 4 * 100, final_step)

  def test_supervisor_run_gan_model_train_ops_multiple_steps(self):
    """Test that the train ops work with the old-style supervisor."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    step = tf.compat.v1.train.create_global_step()
    train_ops = tfgan.GANTrainOps(
        generator_train_op=tf.constant(3.0),
        discriminator_train_op=tf.constant(2.0),
        global_step_inc_op=step.assign_add(1))
    train_steps = tfgan.GANTrainSteps(
        generator_train_steps=3, discriminator_train_steps=4)
    number_of_steps = 1

    # Typical simple Supervisor use.
    train_step_kwargs = {}
    train_step_kwargs['should_stop'] = tf.greater_equal(step, number_of_steps)
    train_step_fn = tfgan.get_sequential_train_steps(train_steps)
    sv = tf.compat.v1.train.Supervisor(logdir='', global_step=step)
    with sv.managed_session(master='') as sess:
      while not sv.should_stop():
        total_loss, should_stop = train_step_fn(
            sess, train_ops, step, train_step_kwargs)
        if should_stop:
          sv.request_stop()
          break

    # Correctness checks.
    self.assertTrue(np.isscalar(total_loss))
    self.assertEqual(17.0, total_loss)

  @parameterized.named_parameters(
      ('gan', create_gan_model),
      ('callable_gan', create_callable_gan_model),
      ('infogan', create_infogan_model),
      ('callable_infogan', create_callable_infogan_model),
      ('acgan', create_acgan_model),
      ('callable_acgan', create_callable_acgan_model),
  )
  def test_train_hooks_exist_in_get_hooks_fn(self, create_gan_model_fn):
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return
    model = create_gan_model_fn()
    loss = tfgan.gan_loss(model)

    g_opt = get_sync_optimizer()
    d_opt = get_sync_optimizer()
    train_ops = tfgan.gan_train_ops(
        model,
        loss,
        g_opt,
        d_opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True)

    sequential_train_hooks = tfgan.get_sequential_train_hooks()(train_ops)
    self.assertLen(sequential_train_hooks, 4)
    sync_opts = [
        hook._sync_optimizer
        for hook in sequential_train_hooks
        if isinstance(hook, get_sync_optimizer_hook_type())
    ]
    self.assertLen(sync_opts, 2)
    self.assertSetEqual(frozenset(sync_opts), frozenset((g_opt, d_opt)))

    joint_train_hooks = tfgan.get_joint_train_hooks()(train_ops)
    self.assertLen(joint_train_hooks, 5)
    sync_opts = [
        hook._sync_optimizer
        for hook in joint_train_hooks
        if isinstance(hook, get_sync_optimizer_hook_type())
    ]
    self.assertLen(sync_opts, 2)
    self.assertSetEqual(frozenset(sync_opts), frozenset((g_opt, d_opt)))


class PatchGANTest(tf.test.TestCase, parameterized.TestCase):
  """Tests that functions work on PatchGAN style output."""

  @parameterized.named_parameters(
      ('gan', create_gan_model),
      ('callable_gan', create_callable_gan_model),
      ('infogan', create_infogan_model),
      ('callable_infogan', create_callable_infogan_model),
      ('acgan', create_acgan_model),
      ('callable_acgan', create_callable_acgan_model),
  )
  def test_patchgan(self, create_gan_model_fn):
    """Ensure that patch-based discriminators work end-to-end."""
    if tf.executing_eagerly():
      # None of the usual utilities work in eager.
      return

    tf.compat.v1.random.set_random_seed(1234)
    model = create_gan_model_fn()
    loss = tfgan.gan_loss(model)

    g_opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    d_opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    train_ops = tfgan.gan_train_ops(model, loss, g_opt, d_opt)

    final_step = tfgan.gan_train(
        train_ops, logdir='', hooks=[tf.estimator.StopAtStepHook(num_steps=2)])
    self.assertTrue(np.isscalar(final_step))
    self.assertEqual(2, final_step)


if __name__ == '__main__':
  tf.test.main()
