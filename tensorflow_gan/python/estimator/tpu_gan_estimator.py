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

"""A TF-GAN-backed GAN Estimator that works on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import inspect

import tensorflow as tf

from tensorflow_gan.python import contrib_utils as contrib
from tensorflow_gan.python import namedtuples as tfgan_tuples
from tensorflow_gan.python import train as tfgan_train
from tensorflow_gan.python.estimator import gan_estimator


__all__ = [
    'TPUGANEstimator',
]

LossFns = collections.namedtuple('_loss_fns', ['g_loss_fn', 'd_loss_fn'])
Optimizers = collections.namedtuple('Optimizers', ['gopt', 'dopt'])


class TPUGANEstimator(tf.contrib.tpu.TPUEstimator):
  """An estimator for Generative Adversarial Networks (GANs) on TPU.

  This Estimator is backed by TFGAN. It is similar to `tfgan.GANEstimator`,
  but works on TPU.

  Example:

  ```python
      import tensorflow as tf
      import tensorflow_gan as tfgan

      # See TFGAN's `train.py` for a description of the generator and
      # discriminator API.
      def generator_fn(generator_inputs):
        ...
        return generated_data

      def discriminator_fn(data, conditioning):
        ...
        return logits

      # Create GAN estimator.
      config = tpu_config.RunConfig(model_dir='/my/dir')
      gan_estimator = tfgan.estimator.TPUGANEstimator(
          generator_fn=generator_fn,
          discriminator_fn=discriminator_fn,
          generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
          discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
          generator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
          discriminator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
          train_batch_size=4,
          config=config)

      # Train estimator.
      gan_estimator.train(train_input_fn, train_steps)

      # Evaluate resulting estimator.
      gan_estimator.evaluate(eval_input_fn, eval_steps)

      # Generate samples from generator.
      predictions = np.array([
          x['generated_data'] for x in gan_estimator.predict(predict_input_fn)])
  ```
  """

  def __init__(self,
               # Arguments to construct the `model_fn`.
               generator_fn=None,
               discriminator_fn=None,
               generator_loss_fn=None,
               discriminator_loss_fn=None,
               generator_optimizer=None,
               discriminator_optimizer=None,
               get_eval_metric_ops_fn=None,
               add_summaries=None,
               joint_train=False,
               gan_train_steps=tfgan_tuples.GANTrainSteps(1, 1),
               # TPUEstimator options.
               model_dir=None,
               config=None,
               params=None,
               use_tpu=True,
               train_batch_size=None,
               eval_batch_size=None,
               predict_batch_size=None,
               batch_axis=None,
               eval_on_tpu=True,
               export_to_tpu=True,
               warm_start_from=None):
    """Initializes a TPUGANEstimator instance.

    Args:
      generator_fn: A python function that takes a Tensor, Tensor list, or
        Tensor dictionary as inputs and returns the outputs of the GAN
        generator. See `TFGAN` for more details and examples. Additionally, if
        it has an argument called `mode`, the Estimator's `mode` will be passed
        in (ex TRAIN, EVAL, PREDICT). This is useful for things like batch
        normalization.
      discriminator_fn: A python function that takes the output of
        `generator_fn` or real data in the GAN setup, and `generator_inputs`.
        Outputs a Tensor in the range [-inf, inf]. See `TFGAN` for more details
        and examples.
      generator_loss_fn: The loss function on the generator. Takes a `GANModel`
        tuple.
      discriminator_loss_fn: The loss function on the discriminator. Takes a
        `GANModel` tuple.
      generator_optimizer: The optimizer for generator updates, or a function
        that takes no arguments and returns an optimizer. This function will
        be called when the default graph is the `GANEstimator`'s graph, so
        utilities like `tf.contrib.framework.get_or_create_global_step` will
        work.
      discriminator_optimizer: Same as `generator_optimizer`, but for the
        discriminator updates.
      get_eval_metric_ops_fn: A function that takes a list of arguments and
        returns a dict of metric results keyed by name. The output of this
        function is passed into `tf.estimator.EstimatorSpec` during evaluation.
        The arguments must be:
            * generator_inputs
            * generated_data
            * real_data
            * discriminator_real_outputs
            * discriminator_gen_outputs
      add_summaries: `None`, a single `SummaryType`, or a list of `SummaryType`.
        This is ignored for jobs that run on TPU, such as the train job if
        `use_tpu` is `True` or the eval job if `eval_on_tpu` is `True`.
      joint_train: A Python boolean. If `True`, jointly train the generator and
        the discriminator. If `False`, sequentially train them. See `train.py`
        in TFGAN for more details on the differences between the two GAN
        training methods.
      gan_train_steps: A `tfgan.GANTrainSteps` named tuple describing the ratio
        of generator to discriminator steps.
      model_dir: Same as `TPUEstimator`: Directory to save model parameters,
        graph and etc. This can also be used to load checkpoints from the
        directory into a estimator to continue training a previously saved
        model. If `None`, the model_dir in `config` will be used if set. If both
        are set, they must be same. If both are `None`, a temporary directory
        will be used.
      config: Same as `TPUEstimator`: An `tpu_config.RunConfig` configuration
        object. Cannot be `None`.
      params: Same as `TPUEstimator`: An optional `dict` of hyper parameters
        that will be passed into `input_fn` and `model_fn`.  Keys are names of
        parameters, values are basic python types. There are reserved keys for
        `TPUEstimator`, including 'batch_size'.
      use_tpu: Same as `TPUEstimator`: A bool indicating whether TPU support is
        enabled. Currently, TPU training and evaluation respect this bit, but
        eval_on_tpu can override execution of eval. See below. Predict still
        happens on CPU.
      train_batch_size: Same as `TPUEstimator`: An int representing the global
        training batch size. TPUEstimator transforms this global batch size to a
        per-shard batch size, as params['batch_size'], when calling `input_fn`
        and `model_fn`. Cannot be `None` if `use_tpu` is `True`. Must be
        divisible by total number of replicas.
      eval_batch_size: Same as `TPUEstimator`: An int representing evaluation
        batch size. Must be divisible by total number of replicas.
      predict_batch_size: Same as `TPUEstimator`: An int representing the
        prediction batch size. Must be divisible by total number of replicas.
      batch_axis: Same as `TPUEstimator`: A python tuple of int values
        describing how each tensor produced by the Estimator `input_fn` should
        be split across the TPU compute shards. For example, if your input_fn
        produced (images, labels) where the images tensor is in `HWCN` format,
        your shard dimensions would be [3, 0], where 3 corresponds to the `N`
        dimension of your images Tensor, and 0 corresponds to the dimension
        along which to split the labels to match up with the corresponding
        images. If None is supplied, and per_host_input_for_training is True,
        batches will be sharded based on the major dimension. If
        tpu_config.per_host_input_for_training is False or `PER_HOST_V2`,
        batch_axis is ignored.
      eval_on_tpu: Same as `TPUEstimator`: If False, evaluation runs on CPU or
        GPU. In this case, the model_fn must return `EstimatorSpec` when called
        with `mode` as `EVAL`.
      export_to_tpu: Same as `TPUEstimator`: If True, `export_savedmodel()`
        exports a metagraph for serving on TPU besides the one on CPU.
      warm_start_from: Same as `TPUEstimator`: Optional string filepath to a
        checkpoint or SavedModel to warm-start from, or a
        `tf.estimator.WarmStartSettings` object to fully configure
        warm-starting.  If the string filepath is provided instead of a
        `WarmStartSettings`, then all variables are warm-started, and it is
        assumed that vocabularies and Tensor names are unchanged.

    Raises:
      ValueError: If loss functions aren't callable.
      ValueError: If `gan_train_steps` isn't a `tfgan_tuples.GANTrainSteps`
        tuple.
      ValueError: If `gan_train_steps` isn't 1:1 training.
    """
    _validate_input_args(
        generator_loss_fn, discriminator_loss_fn, gan_train_steps)
    loss_fns = LossFns(generator_loss_fn, discriminator_loss_fn)
    optimizers = Optimizers(generator_optimizer, discriminator_optimizer)

    # Determine the number of GAN models required to create in order to train
    # in different D:G ratios on TPU.
    required_train_models = _required_train_models(gan_train_steps, joint_train)
    effective_train_batch_size = required_train_models * train_batch_size

    def _model_fn(features, labels, mode, params):
      """GANEstimator model function."""
      del params  # unused
      if mode not in [
          tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
          tf.estimator.ModeKeys.PREDICT
      ]:
        raise ValueError('Mode not recognized: %s' % mode)
      real_data = labels  # rename inputs for clarity
      generator_inputs = features  # rename inputs for clarity

      # Make GANModel, which encapsulates the GAN model architectures.
      # TODO(joelshor): Switch TF-GAN over to TPU-compatible summaries, then
      # remove `add_summaries` logic below.
      is_on_tpu = _is_on_tpu(mode, use_tpu, eval_on_tpu)
      gan_models = _get_gan_models(
          mode,
          generator_fn,
          discriminator_fn,
          real_data,
          generator_inputs,
          num_train_models=required_train_models,
          add_summaries=None if is_on_tpu else add_summaries)

      # Make the TPUEstimatorSpec, which incorporates the model, losses, eval
      # metrics, and optimizers (if required).
      if mode == tf.estimator.ModeKeys.TRAIN:
        estimator_spec = get_train_estimator_spec(
            gan_models, loss_fns, optimizers, joint_train, is_on_tpu,
            gan_train_steps)
      elif mode == tf.estimator.ModeKeys.EVAL:
        estimator_spec = get_eval_estimator_spec(
            gan_models, loss_fns, is_on_tpu, get_eval_metric_ops_fn)
      else:  # predict
        estimator_spec = get_predict_estimator_spec(gan_models)
      assert isinstance(estimator_spec, tf.contrib.tpu.TPUEstimatorSpec)

      return estimator_spec

    super(TPUGANEstimator, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
        use_tpu=use_tpu,
        train_batch_size=effective_train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size,
        batch_axis=batch_axis,
        eval_on_tpu=eval_on_tpu,
        export_to_tpu=export_to_tpu,
        warm_start_from=warm_start_from)


def _validate_input_args(generator_loss_fn, discriminator_loss_fn,
                         gan_train_steps):
  if not callable(generator_loss_fn):
    raise ValueError('generator_loss_fn must be callable.')
  if not callable(discriminator_loss_fn):
    raise ValueError('discriminator_loss_fn must be callable.')
  if not isinstance(gan_train_steps, tfgan_tuples.GANTrainSteps):
    raise ValueError(
        '`gan_train_steps` must be `tfgan_tuples.GANTrainSteps`. Instead, '
        'was type: %s' % type(gan_train_steps))


def _required_train_models(gan_train_steps, joint_train):
  """Returns the required number of train models to create."""
  if joint_train:
    return max(gan_train_steps.generator_train_steps,
               gan_train_steps.discriminator_train_steps)
  else:
    return (gan_train_steps.generator_train_steps +
            gan_train_steps.discriminator_train_steps)


def _get_gan_models(mode,
                    generator_fn,
                    discriminator_fn,
                    real_data,
                    generator_inputs,
                    add_summaries,
                    num_train_models=1,
                    generator_scope='Generator',
                    discriminator_scope='Discriminator'):
  """Makes the GANModel tuple, which encapsulates the GAN model architecture."""
  if mode == tf.estimator.ModeKeys.PREDICT:
    if real_data is not None:
      raise ValueError('`labels` must be `None` when mode is `predict`. '
                       'Instead, found %s' % real_data)
    gan_models = [
        gan_estimator.make_prediction_gan_model(generator_inputs, generator_fn,
                                                generator_scope)
    ]
  else:  # tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.EVAL
    num_models = num_train_models if mode == tf.estimator.ModeKeys.TRAIN else 1
    gan_models = _make_gan_models(
        generator_fn, discriminator_fn, real_data, generator_inputs,
        generator_scope, discriminator_scope, num_models, add_summaries, mode)

  return gan_models


def _slice_data(data, num_slices):
  """Slices data, which can be a list, tuple or dict, into multiple slices.

  Args:
    data: A tensor or a list, tuple or dict of tensors. Nesting is not
      supported. The 0th dimension of all tensors should be divisible by
      `num_slices`.
    num_slices: The number of slices to create.

  Returns:
    A list of length `num_slices`, where each element is of the same type as
    `data`.
  """
  if isinstance(data, (list, tuple)):
    return map(list, zip(*[tf.split(x, num_slices) for x in data]))
  elif isinstance(data, dict):
    dict_of_lists = {k: tf.split(v, num_slices) for k, v in data.items()}
    return [dict(zip(dict_of_lists, x)) for x in zip(*dict_of_lists.values())]
  else:
    return tf.split(data, num_slices)


def _make_gan_models(generator_fn, discriminator_fn, real_data,
                     generator_inputs, generator_scope, discriminator_scope,
                     num_models, add_summaries, mode):
  """Construct one or more `GANModel`s, and optionally pass in `mode`."""
  # If network functions have an argument `mode`, pass mode to it.
  if 'mode' in inspect.getargspec(generator_fn).args:
    generator_fn = functools.partial(generator_fn, mode=mode)
  if 'mode' in inspect.getargspec(discriminator_fn).args:
    discriminator_fn = functools.partial(discriminator_fn, mode=mode)

  real_data_slices = _slice_data(real_data, num_models)
  generator_input_slices = _slice_data(generator_inputs, num_models)

  gan_models = []
  for i in range(num_models):
    gan_model = tfgan_train.gan_model(
        generator_fn,
        discriminator_fn,
        real_data_slices[i],
        generator_input_slices[i],
        generator_scope=generator_scope,
        discriminator_scope=discriminator_scope,
        check_shapes=False)
    gan_models.append(gan_model)

  if add_summaries:
    if not isinstance(add_summaries, (tuple, list)):
      add_summaries = [add_summaries]
    with tf.name_scope(None):
      for summary_type in add_summaries:
        gan_estimator.summary_type_map[summary_type](gan_models[0])

  return gan_models


def _is_on_tpu(mode, use_tpu, eval_on_tpu):
  if mode == tf.estimator.ModeKeys.TRAIN:
    return use_tpu
  elif mode == tf.estimator.ModeKeys.EVAL:
    return eval_on_tpu
  else:
    return False


def get_train_estimator_spec(
    gan_models, loss_fns, optimizers, joint_train, is_on_tpu, gan_train_steps):
  """Estimator spec for train case."""
  gan_losses = _get_losses_for_train(gan_models, loss_fns, is_on_tpu)

  # Construct optimizers if arguments are callable. This has to be done inside
  # the model_fn, since constructable optimizers might create tf.Variables that
  # need to be added to the current tf.Graph.
  optimizers = _maybe_construct_optimizers(optimizers)
  if is_on_tpu:
    optimizers = _maybe_make_cross_shard_optimizers(optimizers)

  scalar_loss = (
      gan_losses[-1].generator_loss + gan_losses[-1].discriminator_loss)
  tpu_train_op = _get_train_op(
      gan_models, gan_losses, optimizers, joint_train, gan_train_steps)

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=tf.estimator.ModeKeys.TRAIN, loss=scalar_loss, train_op=tpu_train_op)


def get_eval_estimator_spec(gan_models, loss_fns, is_on_tpu,
                            get_eval_metric_ops_fn):
  """Estimator spec for eval case."""
  if len(gan_models) > 1:
    raise ValueError('`gan_models` must be length 1 in eval mode. Got length %d'
                     % len(gan_models))
  gan_model = gan_models[0]

  gan_loss = tfgan_tuples.GANLoss(
      generator_loss=loss_fns.g_loss_fn(
          gan_model, add_summaries=not is_on_tpu),
      discriminator_loss=loss_fns.d_loss_fn(
          gan_model, add_summaries=not is_on_tpu))

  # Eval losses for metrics must preserve batch dimension.
  gan_loss_no_reduction = tfgan_tuples.GANLoss(
      generator_loss=loss_fns.g_loss_fn(
          gan_model, add_summaries=False, reduction=tf.losses.Reduction.NONE),
      discriminator_loss=loss_fns.d_loss_fn(
          gan_model, add_summaries=False, reduction=tf.losses.Reduction.NONE))

  # Make the metric function and tensor names.
  if get_eval_metric_ops_fn is not None:
    metric_fn = _make_custom_metric_fn(get_eval_metric_ops_fn)
    tensors_for_metric_fn = _make_custom_metric_tensors(
        gan_model, gan_loss_no_reduction)
  else:
    metric_fn = _make_default_metric_fn()
    tensors_for_metric_fn = _make_default_metric_tensors(gan_loss_no_reduction)

  scalar_loss = gan_loss.generator_loss + gan_loss.discriminator_loss
  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=tf.estimator.ModeKeys.EVAL,
      predictions=gan_model.generated_data,
      loss=scalar_loss,
      eval_metrics=(metric_fn, tensors_for_metric_fn))


def get_predict_estimator_spec(gan_models):
  if len(gan_models) > 1:
    raise ValueError('`gan_models` must be length 1 in predict mode. Got '
                     'length %d' % len(gan_models))
  gan_model = gan_models[0]

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=tf.estimator.ModeKeys.PREDICT,
      predictions={'generated_data': gan_model.generated_data})


def _get_losses_for_train(gan_models, loss_fns, is_on_tpu):
  gan_losses = []
  for gan_model in gan_models:
    gan_losses.append(
        tfgan_tuples.GANLoss(
            generator_loss=loss_fns.g_loss_fn(
                gan_model, add_summaries=not is_on_tpu),
            discriminator_loss=loss_fns.d_loss_fn(
                gan_model, add_summaries=not is_on_tpu)))
  return gan_losses


def _get_train_estimator_spec(gan_models, gan_losses, generator_optimizer,
                              discriminator_optimizer, joint_train,
                              gan_train_steps):
  """Return a TPUEstimatorSpec for the train case."""
  scalar_loss = (
      gan_losses[-1].generator_loss + gan_losses[-1].discriminator_loss)

  def update_ops(substep):
    """Get generator and discriminator update ops for a single training substep.

    We split up the generator and discriminator update ops so that they aren't
    accidentally run multiple times. For now, throw an error if there are update
    ops that aren't associated with either the generator or the discriminator.
    Might modify the `kwargs` dictionary.

    Args:
      substep: An integer index into the substeps of a single global step, made
        up of the joint training, discriminator-only training and generator-only
        training steps.

    Returns:
       A tuple of lists corresponding to
       (generator_update_ops, discriminator_update_ops).
    """
    return tfgan_train._get_update_ops(  # pylint:disable=protected-access
        {}, gan_models[substep].generator_scope.name,
        gan_models[substep].discriminator_scope.name)

  def gen_train_op(substep):
    """Get the generator train op for a single training substep.

    Args:
      substep: An integer index into the substeps of a single global step, made
        up of the joint training, discriminator-only training and generator-only
        training steps.

    Returns:
      An Op that performs a single generator training substep.
    """
    with tf.name_scope('generator_train'):
      return contrib.create_train_op(
          total_loss=gan_losses[substep].generator_loss,
          optimizer=generator_optimizer,
          variables_to_train=gan_models[substep].generator_variables,
          update_ops=update_ops(substep)[0])

  def dis_train_op(substep):
    """Get the discriminator train op for a single training substep.

    Args:
      substep: An integer index into the substeps of a single global step, made
        up of the joint training, discriminator-only training and generator-only
        training steps.

    Returns:
      An Op that performs a single discriminator training substep.
    """
    with tf.name_scope('discriminator_train'):
      return contrib.create_train_op(
          total_loss=gan_losses[substep].discriminator_loss,
          optimizer=discriminator_optimizer,
          variables_to_train=gan_models[substep].discriminator_variables,
          update_ops=update_ops(substep)[1])

  # Either optimize the generator and discriminator sequentially or jointly.
  tpu_train_op = _combine_train_ops(gen_train_op, dis_train_op, joint_train,
                                    gan_train_steps)

  return tf.contrib.tpu.TPUEstimatorSpec(
      loss=scalar_loss, mode=tf.estimator.ModeKeys.TRAIN, train_op=tpu_train_op)


def _get_train_op(gan_models, gan_losses, optimizers, joint_train,
                  gan_train_steps):
  """Return a train op for TPU training."""
  def update_ops(substep):
    """Get generator and discriminator update ops for a single training substep.

    We split up the generator and discriminator update ops so that they aren't
    accidentally run multiple times. For now, throw an error if there are update
    ops that aren't associated with either the generator or the discriminator.
    Might modify the `kwargs` dictionary.

    Args:
      substep: An integer index into the substeps of a single global step, made
        up of the joint training, discriminator-only training and generator-only
        training steps.

    Returns:
       A tuple of lists corresponding to
       (generator_update_ops, discriminator_update_ops).
    """
    return tfgan_train._get_update_ops(  # pylint:disable=protected-access
        {}, gan_models[substep].generator_scope.name,
        gan_models[substep].discriminator_scope.name)

  def gen_train_op(substep):
    """Get the generator train op for a single training substep.

    Args:
      substep: An integer index into the substeps of a single global step, made
        up of the joint training, discriminator-only training and generator-only
        training steps.

    Returns:
      An Op that performs a single generator training substep.
    """
    with tf.name_scope('generator_train'):
      return contrib.create_train_op(
          total_loss=gan_losses[substep].generator_loss,
          optimizer=optimizers.gopt,
          variables_to_train=gan_models[substep].generator_variables,
          update_ops=update_ops(substep)[0])

  def dis_train_op(substep):
    """Get the discriminator train op for a single training substep.

    Args:
      substep: An integer index into the substeps of a single global step, made
        up of the joint training, discriminator-only training and generator-only
        training steps.

    Returns:
      An Op that performs a single discriminator training substep.
    """
    with tf.name_scope('discriminator_train'):
      return contrib.create_train_op(
          total_loss=gan_losses[substep].discriminator_loss,
          optimizer=optimizers.dopt,
          variables_to_train=gan_models[substep].discriminator_variables,
          update_ops=update_ops(substep)[1])

  # Either optimize the generator and discriminator sequentially or jointly.
  return _combine_train_ops(gen_train_op, dis_train_op, joint_train,
                            gan_train_steps)


def _maybe_construct_optimizers(optimizers):
  g_callable = callable(optimizers.gopt)
  gopt = optimizers.gopt() if g_callable  else optimizers.gopt
  d_callable = callable(optimizers.dopt)
  dopt = optimizers.dopt() if d_callable else optimizers.dopt

  return Optimizers(gopt, dopt)


def _combine_train_ops(gen_train_op, dis_train_op, joint_train,
                       gan_train_steps):
  """Combine generator and discriminator train ops into a single op."""
  g_steps = gan_train_steps.generator_train_steps
  d_steps = gan_train_steps.discriminator_train_steps
  joint_steps = 0
  if joint_train:
    joint_steps = min(g_steps, d_steps)
    g_steps -= joint_steps
    d_steps -= joint_steps

  prev_op = tf.no_op()
  for i in range(joint_steps):
    with tf.control_dependencies([prev_op]):
      prev_op = tf.group(
          dis_train_op(i), gen_train_op(i), name='joint_train_%d' % i)
  for i in range(d_steps):
    with tf.control_dependencies([prev_op]):
      prev_op = dis_train_op(i + joint_steps)
  for i in range(g_steps):
    with tf.control_dependencies([prev_op]):
      prev_op = gen_train_op(i + joint_steps + d_steps)

  return prev_op


def _maybe_make_cross_shard_optimizers(optimizers):
  def _maybe_make_cross_shard_optimizer(opt):
    assert not callable(optimizers.gopt)
    if not isinstance(opt, tf.contrib.tpu.CrossShardOptimizer):
      return tf.contrib.tpu.CrossShardOptimizer(opt)
    else:
      return opt

  return Optimizers(_maybe_make_cross_shard_optimizer(optimizers.gopt),
                    _maybe_make_cross_shard_optimizer(optimizers.dopt))


def _make_custom_metric_fn(get_eval_metric_ops_fn):
  """Returns a custom metric function that uses `get_eval_metric_ops_fn`."""
  assert get_eval_metric_ops_fn is not None

  def metric_fn(
      generator_inputs, generated_data, real_data, discriminator_real_outputs,
      discriminator_gen_outputs, generator_loss, discriminator_loss):
    """`metric_fn` used in TPUEstimator to calculate metrics."""
    # Start with the default metrics, then add custom ones.
    eval_metric_ops = _make_default_metric_fn()(
        generator_loss, discriminator_loss)

    custom_eval_metric_ops = get_eval_metric_ops_fn(
        generator_inputs, generated_data, real_data,
        discriminator_real_outputs, discriminator_gen_outputs)
    if not isinstance(custom_eval_metric_ops, dict):
      raise TypeError('`get_eval_metric_ops_fn` must return a dict, '
                      'received: {}'.format(custom_eval_metric_ops))
    eval_metric_ops.update(custom_eval_metric_ops)
    return eval_metric_ops

  return metric_fn


def _make_custom_metric_tensors(gan_model, gan_loss_no_reduction):
  return {
      'generator_loss': gan_loss_no_reduction.generator_loss,
      'discriminator_loss': gan_loss_no_reduction.discriminator_loss,
      'generator_inputs': gan_model.generator_inputs,
      'generated_data': gan_model.generated_data,
      'real_data': gan_model.real_data,
      'discriminator_real_outputs': gan_model.discriminator_real_outputs,
      'discriminator_gen_outputs': gan_model.discriminator_gen_outputs,
  }


def _make_default_metric_fn():
  """Returns the default metric function."""
  def metric_fn(generator_loss, discriminator_loss):
    return {
        'generator_loss': tf.metrics.mean(generator_loss),
        'discriminator_loss': tf.metrics.mean(discriminator_loss),
    }
  return metric_fn


def _make_default_metric_tensors(gan_loss_no_reduction):
  return {
      'generator_loss': gan_loss_no_reduction.generator_loss,
      'discriminator_loss': gan_loss_no_reduction.discriminator_loss,
  }
