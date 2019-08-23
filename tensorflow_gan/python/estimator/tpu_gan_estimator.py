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


class TPUGANEstimator(tf.compat.v1.estimator.tpu.TPUEstimator):
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

  def __init__(
      self,
      # Arguments to construct the `model_fn`.
      generator_fn=None,
      discriminator_fn=None,
      generator_loss_fn=None,
      discriminator_loss_fn=None,
      generator_optimizer=None,
      discriminator_optimizer=None,
      prepare_arguments_for_eval_metric_fn=None,
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
        utilities like `tf.train.get_or_create_global_step` will
        work.
      discriminator_optimizer: Same as `generator_optimizer`, but for the
        discriminator updates.
      prepare_arguments_for_eval_metric_fn: A function that takes a list of
        arguments and returns a nested structure of tensors keyed by name. The
        returned tensors must be compatible with TPUEstimatorSpec.eval_metrics
        (i.e., in batch-major format, where the batch size is the first
        dimension) and will be passed to the provided get_eval_metric_ops_fn.
        The arguments must be:
            * generator_inputs
            * generated_data
            * real_data
            * discriminator_real_outputs
            * discriminator_gen_outputs
        The default impelementation simply returns the arguments as-is. This
        function is executed on the TPU, allowing for compute-heavy eval-only
        operations to be performed.
      get_eval_metric_ops_fn: A function that takes a list of arguments and
        returns a dict of metric results keyed by name, exectuted on CPU. The
        arguments of the function should be the keys of the dict returned
        by prepare_arguments_for_eval_metric_fn (see the
        prepare_arguments_for_eval_metric_fn for the defaults), and should
        return a dict from metric string name to the result of calling a metric
        function, namely a (metric_tensor, update_op) tuple.
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
        `TPUEstimator`, including 'batch_size'. If any `params` are args to
        TF-GAN's `gan_loss`, they will be passed to `gan_loss` during training
        and evaluation.
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
      if mode not in [
          tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
          tf.estimator.ModeKeys.PREDICT
      ]:
        raise ValueError('Mode not recognized: %s' % mode)
      real_data = labels  # rename inputs for clarity
      generator_inputs = features  # rename inputs for clarity

      # Collect GANModel builder functions, which encapsulate the GAN model
      # architectures. Don't actually execute them here, since the functions
      # actually create the TF ops and the variable reads need to be chained
      # after the writes from the previous step. Instead just pass the functions
      # with bound arguments down so that they can easily be executed later.
      gan_model_fns = _get_gan_model_fns(
          mode,
          generator_fn,
          discriminator_fn,
          real_data,
          generator_inputs,
          num_train_models=required_train_models)

      # TODO(joelshor): Switch TF-GAN over to TPU-compatible summaries, then
      # remove `add_summaries` logic below.
      is_on_tpu = _is_on_tpu(mode, use_tpu, eval_on_tpu)
      summary_types = None if is_on_tpu else add_summaries

      # Make the TPUEstimatorSpec, which incorporates the model, losses, eval
      # metrics, and optimizers (if required).
      gan_loss_kwargs = gan_estimator.extract_gan_loss_args_from_params(params)
      if mode == tf.estimator.ModeKeys.TRAIN:
        estimator_spec = get_train_estimator_spec(
            gan_model_fns,
            loss_fns,
            gan_loss_kwargs,
            optimizers,
            joint_train,
            is_on_tpu,
            gan_train_steps,
            add_summaries=summary_types)
      elif mode == tf.estimator.ModeKeys.EVAL:
        estimator_spec = get_eval_estimator_spec(
            gan_model_fns,
            loss_fns,
            gan_loss_kwargs,
            prepare_arguments_for_eval_metric_fn,
            get_eval_metric_ops_fn,
            add_summaries=summary_types)
      else:  # predict
        estimator_spec = get_predict_estimator_spec(gan_model_fns)
      assert isinstance(estimator_spec,
                        tf.compat.v1.estimator.tpu.TPUEstimatorSpec)

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


def _get_gan_model_fns(mode,
                       generator_fn,
                       discriminator_fn,
                       real_data,
                       generator_inputs,
                       num_train_models=1,
                       generator_scope='Generator',
                       discriminator_scope='Discriminator'):
  """Makes the GANModel tuple, which encapsulates the GAN model architecture."""
  if mode == tf.estimator.ModeKeys.PREDICT:
    if real_data is not None:
      raise ValueError('`labels` must be `None` when mode is `predict`. '
                       'Instead, found %s' % real_data)
    gan_models = [
        functools.partial(gan_estimator.make_prediction_gan_model,
                          generator_inputs, generator_fn, generator_scope)
    ]
  else:  # tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.EVAL
    num_models = num_train_models if mode == tf.estimator.ModeKeys.TRAIN else 1
    gan_models = _make_gan_model_fns(generator_fn, discriminator_fn, real_data,
                                     generator_inputs, generator_scope,
                                     discriminator_scope, num_models, mode)

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


def _make_gan_model_fns(generator_fn, discriminator_fn, real_data,
                        generator_inputs, generator_scope, discriminator_scope,
                        num_models, mode):
  """Construct one or more no-arg functions that construct `GANModel`s."""
  # If network functions have an argument `mode`, pass mode to it.
  if 'mode' in inspect.getargspec(generator_fn).args:
    generator_fn = functools.partial(generator_fn, mode=mode)
  if 'mode' in inspect.getargspec(discriminator_fn).args:
    discriminator_fn = functools.partial(discriminator_fn, mode=mode)

  real_data_slices = _slice_data(real_data, num_models)
  generator_input_slices = _slice_data(generator_inputs, num_models)

  gan_model_fns = []
  for i in range(num_models):
    gan_model_fn = functools.partial(
        tfgan_train.gan_model,
        generator_fn,
        discriminator_fn,
        real_data_slices[i],
        generator_input_slices[i],
        generator_scope=generator_scope,
        discriminator_scope=discriminator_scope,
        check_shapes=False)
    gan_model_fns.append(gan_model_fn)

  return gan_model_fns


def _is_on_tpu(mode, use_tpu, eval_on_tpu):
  if mode == tf.estimator.ModeKeys.TRAIN:
    return use_tpu
  elif mode == tf.estimator.ModeKeys.EVAL:
    return eval_on_tpu
  else:
    return False


def _maybe_add_summaries(gan_model, add_summaries):
  """Maybe add summaries."""
  if add_summaries:
    if not isinstance(add_summaries, (tuple, list)):
      add_summaries = [add_summaries]
    with tf.compat.v1.name_scope(''):  # Clear name scope.
      for summary_type in add_summaries:
        gan_estimator.summary_type_map[summary_type](gan_model)


def get_train_estimator_spec(gan_model_fns, loss_fns, gan_loss_kwargs,
                             optimizers, joint_train, is_on_tpu,
                             gan_train_steps, add_summaries):
  """Estimator spec for train case."""
  # Construct optimizers if arguments are callable. This has to be done inside
  # the model_fn, since constructable optimizers might create tf.Variables that
  # need to be added to the current tf.Graph.
  optimizers = _maybe_construct_optimizers(optimizers)
  if is_on_tpu:
    optimizers = _maybe_make_cross_shard_optimizers(optimizers)

  tpu_train_op, scalar_loss = _get_train_op(
      gan_model_fns, loss_fns, gan_loss_kwargs, optimizers, joint_train,
      gan_train_steps, add_summaries)

  return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
      mode=tf.estimator.ModeKeys.TRAIN, loss=scalar_loss, train_op=tpu_train_op)


def _predictions_from_generator_output(generated_data):
  """Returns a predictions tensor from generator output of unknown structure.

  If `generated_data` is a Tensor, just return it.
  If `generated_data` is a list or tuple, return the first element.
  If `generated_data` is a dictionary, just return it.

  Args:
    generated_data: Output of generator.

  Returns:
    A single Tensor.
  """

  if isinstance(generated_data, tf.Tensor):
    return generated_data
  elif isinstance(generated_data, (list, tuple)):
    return generated_data[0]
  elif isinstance(generated_data, dict):
    return generated_data
  else:
    raise ValueError(
        'Generator produced output of type %s, but TPUGANEstimator cannot make '
        'predictions from it.' % type(generated_data))


def get_eval_estimator_spec(gan_model_fns, loss_fns, gan_loss_kwargs,
                            prepare_arguments_for_eval_metric_fn,
                            get_eval_metric_ops_fn, add_summaries):
  """Estimator spec for eval case."""
  assert len(gan_model_fns) == 1, (
      '`gan_models` must be length 1 in eval mode. Got length %d' %
      len(gan_model_fns))

  gan_model = gan_model_fns[0]()

  _maybe_add_summaries(gan_model, add_summaries)

  # Eval losses for metrics must preserve batch dimension.
  kwargs = gan_loss_kwargs or {}
  gan_loss_no_reduction = tfgan_train.gan_loss(
      gan_model,
      loss_fns.g_loss_fn,
      loss_fns.d_loss_fn,
      add_summaries=add_summaries,
      reduction=tf.compat.v1.losses.Reduction.NONE,
      **kwargs)

  if prepare_arguments_for_eval_metric_fn is None:
    # Set the default prepare_arguments_for_eval_metric_fn value: a function
    # that returns its arguments in a dict.
    prepare_arguments_for_eval_metric_fn = lambda **kwargs: kwargs

  default_metric_fn = _make_default_metric_fn()
  # Prepare tensors needed for calculating the metrics: the first element in
  # `tensors_for_metric_fn` holds a dict containing the arguments for
  # `default_metric_fn`, and the second element holds a dict for arguments for
  # `get_eval_metric_ops_fn` (if it is not None).
  tensors_for_metric_fn = [_make_default_metric_tensors(gan_loss_no_reduction)]
  if get_eval_metric_ops_fn is not None:
    tensors_for_metric_fn.append(prepare_arguments_for_eval_metric_fn(
        **_make_custom_metric_tensors(gan_model)))

  scalar_loss = tf.compat.v1.losses.compute_weighted_loss(
      gan_loss_no_reduction.discriminator_loss,
      loss_collection=None,
      reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

  # TPUEstimatorSpec.eval_metrics expects a function and a list of tensors,
  # however, some sturctures in tensors_for_metric_fn might be dictionaries
  # (e.g., generator_inputs and real_data). We therefore need to flatten
  # tensors_for_metric_fn before passing them to the function and then restoring
  # the original structure inside the function.
  def _metric_fn_wrapper(*args):
    """Unflattens the arguments and pass them to the metric functions."""
    unpacked_arguments = tf.nest.pack_sequence_as(tensors_for_metric_fn, args)
    # Calculate default metrics.
    metrics = default_metric_fn(**unpacked_arguments[0])
    if get_eval_metric_ops_fn is not None:
      # Append custom metrics.
      custom_eval_metric_ops = get_eval_metric_ops_fn(**unpacked_arguments[1])
      if not isinstance(custom_eval_metric_ops, dict):
        raise TypeError('`get_eval_metric_ops_fn` must return a dict, '
                        'received: {}'.format(custom_eval_metric_ops))
      metrics.update(custom_eval_metric_ops)

    return metrics

  flat_tensors = tf.nest.flatten(tensors_for_metric_fn)
  if not all(isinstance(t, tf.Tensor) for t in flat_tensors):
    raise ValueError('All objects nested within the TF-GAN model must be '
                     'tensors. Instead, types are: %s.' %
                     str([type(v) for v in flat_tensors]))
  return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
      mode=tf.estimator.ModeKeys.EVAL,
      predictions=_predictions_from_generator_output(gan_model.generated_data),
      loss=scalar_loss,
      eval_metrics=(_metric_fn_wrapper, flat_tensors))


def get_predict_estimator_spec(gan_model_fns):
  assert len(gan_model_fns) == 1, (
      '`gan_models` must be length 1 in predict mode. Got length %d' %
      len(gan_model_fns))

  gan_model = gan_model_fns[0]()

  preds = _predictions_from_generator_output(gan_model.generated_data)
  return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
      mode=tf.estimator.ModeKeys.PREDICT, predictions={'generated_data': preds})


def _get_loss_for_train(gan_model, loss_fns, gan_loss_kwargs, add_summaries):
  kwargs = gan_loss_kwargs or {}
  return tfgan_train.gan_loss(
      gan_model,
      loss_fns.g_loss_fn,
      loss_fns.d_loss_fn,
      add_summaries=add_summaries,
      **kwargs)


def _get_train_op(gan_model_fns, loss_fns, gan_loss_kwargs, optimizers,
                  joint_train, gan_train_steps, add_summaries):
  """Return a train op for TPU training."""

  def update_ops(gan_model):
    """Get generator and discriminator update ops for a single training substep.

    We split up the generator and discriminator update ops so that they aren't
    accidentally run multiple times. For now, throw an error if there are update
    ops that aren't associated with either the generator or the discriminator.
    Might modify the `kwargs` dictionary.

    Args:
      gan_model: The GANModel tuple.

    Returns:
       A tuple of lists corresponding to
       (generator_update_ops, discriminator_update_ops).
    """
    return tfgan_train._get_update_ops(  # pylint:disable=protected-access
        {}, gan_model.generator_scope.name, gan_model.discriminator_scope.name)

  def gen_train_op(gan_model, gan_loss):
    """Get the generator train op for a single training substep.

    Args:
      gan_model: The GANModel tuple.
      gan_loss: The GANLoss tuple.

    Returns:
      An Op that performs a single generator training substep.
    """
    with tf.compat.v1.name_scope('generator_train'):
      return contrib.create_train_op(
          total_loss=gan_loss.generator_loss,
          optimizer=optimizers.gopt,
          variables_to_train=gan_model.generator_variables,
          global_step=None,
          update_ops=update_ops(gan_model)[0])

  def dis_train_op(gan_model, gan_loss):
    """Get the discriminator train op for a single training substep.

    Args:
      gan_model: The GANModel tuple.
      gan_loss: The GANLoss tuple.

    Returns:
      An Op that performs a single discriminator training substep.
    """
    with tf.compat.v1.name_scope('discriminator_train'):
      return contrib.create_train_op(
          total_loss=gan_loss.discriminator_loss,
          optimizer=optimizers.dopt,
          variables_to_train=gan_model.discriminator_variables,
          global_step=None,
          update_ops=update_ops(gan_model)[1])

  # Either optimize the generator and discriminator sequentially or jointly.
  g_steps = gan_train_steps.generator_train_steps
  d_steps = gan_train_steps.discriminator_train_steps
  joint_steps = 0
  if joint_train:
    joint_steps = min(g_steps, d_steps)
    g_steps -= joint_steps
    d_steps -= joint_steps
  total_steps = joint_steps + d_steps + g_steps

  prev_op = tf.no_op()
  scalar_loss = 0
  for i in range(total_steps):
    # For each substep, make sure that the forward pass ops are created with
    # control dependencies on the train op of the previous substep. We can't
    # just chain the train ops because the weight read for substep n will end up
    # happening before the weights are updated in substep n-1.
    with tf.control_dependencies([prev_op]):
      gan_model = gan_model_fns[i]()
      _maybe_add_summaries(gan_model, add_summaries and i == total_steps - 1)
      gan_loss = _get_loss_for_train(gan_model, loss_fns, gan_loss_kwargs,
                                     add_summaries)
      scalar_loss = gan_loss.discriminator_loss
      if i < joint_steps:
        prev_op = tf.group(
            dis_train_op(gan_model, gan_loss),
            gen_train_op(gan_model, gan_loss),
            name='joint_train_%d' % i)
      elif i < joint_steps + d_steps:
        prev_op = dis_train_op(gan_model, gan_loss)
      else:
        prev_op = gen_train_op(gan_model, gan_loss)

  with tf.control_dependencies([prev_op]):
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_op = global_step.assign_add(1)

  return train_op, scalar_loss


def _maybe_construct_optimizers(optimizers):
  g_callable = callable(optimizers.gopt)
  gopt = optimizers.gopt() if g_callable else optimizers.gopt
  d_callable = callable(optimizers.dopt)
  dopt = optimizers.dopt() if d_callable else optimizers.dopt

  return Optimizers(gopt, dopt)


def _maybe_make_cross_shard_optimizers(optimizers):
  def _maybe_make_cross_shard_optimizer(opt):
    assert not callable(optimizers.gopt)
    if not isinstance(opt, tf.compat.v1.tpu.CrossShardOptimizer):
      return tf.compat.v1.tpu.CrossShardOptimizer(opt)
    else:
      return opt

  return Optimizers(_maybe_make_cross_shard_optimizer(optimizers.gopt),
                    _maybe_make_cross_shard_optimizer(optimizers.dopt))


def _make_custom_metric_tensors(gan_model):
  return {
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
        'generator_loss': tf.compat.v1.metrics.mean(generator_loss),
        'discriminator_loss': tf.compat.v1.metrics.mean(discriminator_loss),
    }
  return metric_fn


def _make_default_metric_tensors(gan_loss_no_reduction):
  return {
      'generator_loss': gan_loss_no_reduction.generator_loss,
      'discriminator_loss': gan_loss_no_reduction.discriminator_loss,
  }
