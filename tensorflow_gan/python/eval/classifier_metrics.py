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

"""Classifier-based evaluation tools for TF-GAN.

These methods come from https://arxiv.org/abs/1606.03498,
https://arxiv.org/abs/1706.08500, and https://arxiv.org/abs/1801.01401.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_gan.python.eval import eval_utils
import tensorflow_probability as tfp

__all__ = [
    'run_classifier_fn',
    'sample_and_run_classifier_fn',
    'classifier_score',
    'classifier_score_streaming',
    'classifier_score_from_logits',
    'classifier_score_from_logits_streaming',
    'frechet_classifier_distance',
    'frechet_classifier_distance_streaming',
    'frechet_classifier_distance_from_activations',
    'frechet_classifier_distance_from_activations_streaming',
    'mean_only_frechet_classifier_distance_from_activations',
    'diagonal_only_frechet_classifier_distance_from_activations',
    'kernel_classifier_distance',
    'kernel_classifier_distance_and_std',
    'kernel_classifier_distance_from_activations',
    'kernel_classifier_distance_and_std_from_activations',
]


def run_classifier_fn(input_tensor,
                      classifier_fn,
                      num_batches=1,
                      dtypes=None,
                      name='RunClassifierFn'):
  """Runs a network from a TF-Hub module.

  If there are multiple outputs, cast them to tf.float32.

  Args:
    input_tensor: Input tensors.
    classifier_fn: A function that takes a single argument and returns the
      outputs of the classifier. If `num_batches` is greater than 1, the
      structure of the outputs of `classifier_fn` must match the structure of
      `dtypes`.
    num_batches: Number of batches to split `tensor` in to in order to
      efficiently run them through the classifier network. This is useful if
      running a large batch would consume too much memory, but running smaller
      batches is feasible.
    dtypes: If `classifier_fn` returns more than one element or `num_batches` is
      greater than 1, `dtypes` must have the same structure as the return value
      of `classifier_fn` but with each output replaced by the expected dtype of
      the output. If `classifier_fn` returns on element or `num_batches` is 1,
      then `dtype` can be `None.
    name: Name scope for classifier.

  Returns:
    The output of the module, or just `outputs`.

  Raises:
    ValueError: If `classifier_fn` return multiple outputs but `dtypes` isn't
      specified, or is incorrect.
  """
  if num_batches > 1:
    # Compute the classifier splits using the memory-efficient `map_fn`.
    input_list = tf.split(input_tensor, num_or_size_splits=num_batches)
    classifier_outputs = tf.map_fn(
        fn=classifier_fn,
        elems=tf.stack(input_list),
        dtype=dtypes,
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name=name)
    classifier_outputs = tf.nest.map_structure(
        lambda x: tf.concat(tf.unstack(x), 0), classifier_outputs)
  else:
    classifier_outputs = classifier_fn(input_tensor)

  return classifier_outputs


def sample_and_run_classifier_fn(sample_fn,
                                 sample_inputs,
                                 classifier_fn,
                                 dtypes=None,
                                 name='SampleAndRunClassifierFn'):
  """Sampes Tensors from distribution then runs them through a function.

  This is the same as `sample_and_run_image_classifier`, but instead of taking
  a classifier GraphDef it takes a function.

  If there are multiple outputs, cast them to tf.float32.

  NOTE: Running the sampler can affect the original weights if, for instance,
  there are assign ops in the sampler. See
  `test_assign_variables_in_sampler_runs` in the unit tests for an example.

  Args:
    sample_fn: A function that takes a single argument and returns images. This
      function samples from an image distribution.
    sample_inputs: A list of inputs to pass to `sample_fn`.
    classifier_fn: A function that takes a single argument and returns the
      outputs of the classifier. If `num_batches` is greater than 1, the
      structure of the outputs of `classifier_fn` must match the structure of
      `dtypes`.
    dtypes: If `classifier_fn` returns more than one element or `num_batches` is
      greater than 1, `dtypes` must have the same structure as the return value
      of `classifier_fn` but with each output replaced by the expected dtype of
      the output. If `classifier_fn` returns on element or `num_batches` is 1,
      then `dtype` can be `None.
    name: Name scope for classifier.

  Returns:
    Classifier output if `output_tensor` is a string, or a list of outputs if
    `output_tensor` is a list.

  Raises:
    ValueError: If `classifier_fn` return multiple outputs but `dtypes` isn't
      specified, or is incorrect.
  """
  def _fn(x):
    tensor = sample_fn(x)
    return classifier_fn(tensor)
  if len(sample_inputs) > 1:
    classifier_outputs = tf.map_fn(
        fn=_fn,
        elems=tf.stack(sample_inputs),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        dtype=dtypes,
        name=name)
    classifier_outputs = tf.nest.map_structure(
        lambda x: tf.concat(tf.unstack(x), 0), classifier_outputs)
  else:
    classifier_outputs = _fn(sample_inputs[0])

  return classifier_outputs


def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Compute square root of a symmetric matrix.

  Note that this is different from an elementwise square root. We want to
  compute M' where M' = sqrt(mat) such that M' * M' = mat.

  Also note that this method **only** works for symmetric matrices.

  Args:
    mat: Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
      rooted to guard against numerical instability.

  Returns:
    Matrix square root of mat.
  """
  # Unlike numpy, tensorflow's return order is (s, u, v)
  s, u, v = tf.linalg.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
  # Note that the v returned by Tensorflow is v = V
  # (when referencing the equation A = U S V^T)
  # This is unlike Numpy which returns v = V^T
  return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)


def kl_divergence(p, p_logits, q):
  """Computes the Kullback-Liebler divergence between p and q.

  This function uses p's logits in some places to improve numerical stability.

  Specifically:

  KL(p || q) = sum[ p * log(p / q) ]
    = sum[ p * ( log(p)                - log(q) ) ]
    = sum[ p * ( log_softmax(p_logits) - log(q) ) ]

  Args:
    p: A 2-D floating-point Tensor p_ij, where `i` corresponds to the minibatch
      example and `j` corresponds to the probability of being in class `j`.
    p_logits: A 2-D floating-point Tensor corresponding to logits for `p`.
    q: A 1-D floating-point Tensor, where q_j corresponds to the probability of
      class `j`.

  Returns:
    KL divergence between two distributions. Output dimension is 1D, one entry
    per distribution in `p`.

  Raises:
    ValueError: If any of the inputs aren't floating-point.
    ValueError: If p or p_logits aren't 2D.
    ValueError: If q isn't 1D.
  """
  for tensor in [p, p_logits, q]:
    if not tensor.dtype.is_floating:
      tensor_name = tensor if tf.executing_eagerly() else tensor.name
      raise ValueError('Input %s must be floating type.' % tensor_name)
  p.shape.assert_has_rank(2)
  p_logits.shape.assert_has_rank(2)
  q.shape.assert_has_rank(1)
  return tf.reduce_sum(
      input_tensor=p * (tf.nn.log_softmax(p_logits) - tf.math.log(q)), axis=1)


def _classifier_score_helper(input_tensor,
                             classifier_fn,
                             num_batches=1,
                             streaming=False):
  """A helper function for evaluating the classifier score."""
  if num_batches > 1:
    # Compute the classifier splits using the memory-efficient `map_fn`.
    input_list = tf.split(input_tensor, num_or_size_splits=num_batches)
    logits = tf.map_fn(
        fn=classifier_fn,
        elems=tf.stack(input_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    logits = tf.concat(tf.unstack(logits), 0)
  else:
    logits = classifier_fn(input_tensor)

  return _classifier_score_from_logits_helper(logits, streaming=streaming)


def classifier_score(input_tensor, classifier_fn, num_batches=1):
  """Classifier score for evaluating a conditional generative model.

  This is based on the Inception Score, but for an arbitrary classifier.

  This technique is described in detail in https://arxiv.org/abs/1606.03498. In
  summary, this function calculates

  exp( E[ KL(p(y|x) || p(y)) ] )

  which captures how different the network's classification prediction is from
  the prior distribution over classes.

  NOTE: This function consumes input tensors, computes their logits, and then
  computes the classifier score. If you would like to precompute many logits for
  large batches, use classifier_score_from_logits(), which this method also
  uses.

  Args:
    input_tensor: Input to the classifier function.
    classifier_fn: A function that takes tensors and produces logits based on a
      classifier.
    num_batches: Number of batches to split `generated_images` in to in order to
      efficiently run them through the classifier network.

  Returns:
    The classifier score. A floating-point scalar of the same type as the output
    of `classifier_fn`.
  """
  return _classifier_score_helper(
      input_tensor, classifier_fn, num_batches, streaming=False)


def classifier_score_streaming(input_tensor, classifier_fn, num_batches=1):
  """A streaming version of classifier_score.

  Keeps an internal state that continuously tracks the score. This internal
  state should be initialized with tf.initializers.local_variables().

  Args:
    input_tensor: Input to the classifier function.
    classifier_fn: A function that takes tensors and produces logits based on a
      classifier.
    num_batches: Number of batches to split `generated_images` in to in order to
      efficiently run them through the classifier network.

  Returns:
    A tuple containing the classifier score and a tf.Operation. The tf.Operation
    has the same value as the score, and has an additional side effect of
    updating the internal state with the given tensors.
  """
  return _classifier_score_helper(
      input_tensor, classifier_fn, num_batches, streaming=True)


def _classifier_score_from_logits_helper(logits, streaming=False):
  """A helper function for evaluating the classifier score from logits."""
  logits = tf.convert_to_tensor(value=logits)
  logits.shape.assert_has_rank(2)

  # Use maximum precision for best results.
  logits_dtype = logits.dtype
  if logits_dtype != tf.float64:
    logits = tf.cast(logits, tf.float64)

  p = tf.nn.softmax(logits)
  if streaming:
    # Note: The following streaming mean operation assumes all instances of
    # logits have the same batch size.
    q_ops = eval_utils.streaming_mean_tensor_float64(
        tf.reduce_mean(input_tensor=p, axis=0))
    # kl = kl_divergence(p, logits, q)
    # = tf.reduce_sum(p * (tf.nn.log_softmax(logits) - tf.math.log(q)), axis=1)
    # = tf.reduce_sum(p * tf.nn.log_softmax(logits), axis=1)
    #   - tf.reduce_sum(p * tf.math.log(q), axis=1)
    # log_score = tf.reduce_mean(kl)
    # = tf.reduce_mean(tf.reduce_sum(p * tf.nn.log_softmax(logits), axis=1))
    #   - tf.reduce_mean(tf.reduce_sum(p * tf.math.log(q), axis=1))
    # = tf.reduce_mean(tf.reduce_sum(p * tf.nn.log_softmax(logits), axis=1))
    #   - tf.reduce_sum(tf.reduce_mean(p, axis=0) * tf.math.log(q))
    # = tf.reduce_mean(tf.reduce_sum(p * tf.nn.log_softmax(logits), axis=1))
    #   - tf.reduce_sum(q * tf.math.log(q))
    plogp_mean_ops = eval_utils.streaming_mean_tensor_float64(
        tf.reduce_mean(
            input_tensor=tf.reduce_sum(
                input_tensor=p * tf.nn.log_softmax(logits), axis=1)))
    log_score_ops = tuple(
        plogp_mean_val - tf.reduce_sum(input_tensor=q_val * tf.math.log(q_val))
        for plogp_mean_val, q_val in zip(plogp_mean_ops, q_ops))
  else:
    q = tf.reduce_mean(input_tensor=p, axis=0)
    kl = kl_divergence(p, logits, q)
    kl.shape.assert_has_rank(1)
    log_score_ops = (tf.reduce_mean(input_tensor=kl),)
  # log_score_ops contains the score value and possibly the update_op. We
  # apply the same operation on all its elements to make sure their value is
  # consistent.
  final_score_tuple = tuple(tf.exp(value) for value in log_score_ops)
  if logits_dtype != tf.float64:
    final_score_tuple = tuple(
        tf.cast(value, logits_dtype) for value in final_score_tuple)

  if streaming:
    return final_score_tuple
  else:
    return final_score_tuple[0]


def classifier_score_from_logits(logits):
  """Classifier score for evaluating a generative model from logits.

  This method computes the classifier score for a set of logits. This can be
  used independently of the classifier_score() method, especially in the case
  of using large batches during evaluation where we would like precompute all
  of the logits before computing the classifier score.

  This technique is described in detail in https://arxiv.org/abs/1606.03498. In
  summary, this function calculates:

  exp( E[ KL(p(y|x) || p(y)) ] )

  which captures how different the network's classification prediction is from
  the prior distribution over classes.

  Args:
    logits: Precomputed 2D tensor of logits that will be used to compute the
      classifier score.

  Returns:
    The classifier score. A floating-point scalar of the same type as the output
    of `logits`.
  """
  return _classifier_score_from_logits_helper(logits, streaming=False)


def classifier_score_from_logits_streaming(logits):
  """A streaming version of classifier_score_from_logits.

  Keeps an internal state that continuously tracks the score. This internal
  state should be initialized with tf.initializers.local_variables().

  Args:
    logits: Precomputed 2D tensor of logits that will be used to compute the
      classifier score.

  Returns:
    A tuple containing the classifier score and a tf.Operation. The tf.Operation
    has the same value as the score, and has an additional side effect of
    updating the internal state with the given tensors.
  """
  return _classifier_score_from_logits_helper(logits, streaming=True)


def trace_sqrt_product(sigma, sigma_v):
  """Find the trace of the positive sqrt of product of covariance matrices.

  '_symmetric_matrix_square_root' only works for symmetric matrices, so we
  cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
  ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

  Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
  We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
  Note the following properties:
  (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
     => eigenvalues(A A B B) = eigenvalues (A B B A)
  (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
     => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
  (iii) forall M: trace(M) = sum(eigenvalues(M))
     => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                   = sum(sqrt(eigenvalues(A B B A)))
                                   = sum(eigenvalues(sqrt(A B B A)))
                                   = trace(sqrt(A B B A))
                                   = trace(sqrt(A sigma_v A))
  A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
  use the _symmetric_matrix_square_root function to find the roots of these
  matrices.

  Args:
    sigma: a square, symmetric, real, positive semi-definite covariance matrix
    sigma_v: same as sigma

  Returns:
    The trace of the positive square root of sigma*sigma_v
  """

  # Note sqrt_sigma is called "A" in the proof above
  sqrt_sigma = _symmetric_matrix_square_root(sigma)

  # This is sqrt(A sigma_v A) above
  sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

  return tf.linalg.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def _frechet_classifier_distance_helper(input_tensor1,
                                        input_tensor2,
                                        classifier_fn,
                                        num_batches=1,
                                        streaming=False):
  """A helper function for evaluating the frechet classifier distance."""
  input_list1 = tf.split(input_tensor1, num_or_size_splits=num_batches)
  input_list2 = tf.split(input_tensor2, num_or_size_splits=num_batches)

  stack1 = tf.stack(input_list1)
  stack2 = tf.stack(input_list2)

  # Compute the activations using the memory-efficient `map_fn`.
  def compute_activations(elems):
    return tf.map_fn(
        fn=classifier_fn,
        elems=elems,
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')

  activations1 = compute_activations(stack1)
  activations2 = compute_activations(stack2)

  # Ensure the activations have the right shapes.
  activations1 = tf.concat(tf.unstack(activations1), 0)
  activations2 = tf.concat(tf.unstack(activations2), 0)

  return _frechet_classifier_distance_from_activations_helper(
      activations1, activations2, streaming=streaming)


def frechet_classifier_distance(input_tensor1,
                                input_tensor2,
                                classifier_fn,
                                num_batches=1):
  """Classifier distance for evaluating a generative model.

  This is based on the Frechet Inception distance, but for an arbitrary
  classifier.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calculates

              |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute Frechet classifier distance when comparing two
  generative models.

  NOTE: This function consumes inputs, computes their activations, and then
  computes the classifier score. If you would like to precompute many
  activations for large batches, please use
  frechet_clasifier_distance_from_activations(), which this method also uses.

  Args:
    input_tensor1: First tensor to use as inputs.
    input_tensor2: Second tensor to use as inputs.
    classifier_fn: A function that takes tensors and produces activations based
      on a classifier.
    num_batches: Number of batches to split images in to in order to efficiently
      run them through the classifier network.

  Returns:
    The Frechet Inception distance. A floating-point scalar of the same type
    as the output of `classifier_fn`.
  """
  return _frechet_classifier_distance_helper(
      input_tensor1,
      input_tensor2,
      classifier_fn,
      num_batches,
      streaming=False)


def frechet_classifier_distance_streaming(input_tensor1,
                                          input_tensor2,
                                          classifier_fn,
                                          num_batches=1):
  """A streaming version of frechet_classifier_distance.

  Keeps an internal state that continuously tracks the score. This internal
  state should be initialized with tf.initializers.local_variables().

  Args:
    input_tensor1: First tensor to use as inputs.
    input_tensor2: Second tensor to use as inputs.
    classifier_fn: A function that takes tensors and produces activations based
      on a classifier.
    num_batches: Number of batches to split images in to in order to efficiently
      run them through the classifier network.

  Returns:
    A tuple containing the classifier score and a tf.Operation. The tf.Operation
    has the same value as the score, and has an additional side effect of
    updating the internal state with the given tensors.
  """
  return _frechet_classifier_distance_helper(
      input_tensor1,
      input_tensor2,
      classifier_fn,
      num_batches,
      streaming=True)


def mean_only_frechet_classifier_distance_from_activations(
    activations1, activations2):
  """Classifier distance for evaluating a generative model from activations.

  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calcuates

                                |m - m_w|^2

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute frechet classifier distance when comparing two
  generative models.

  In this variant, we only compute the difference between the means of the
  fitted Gaussians. The computation leads to O(n) vs. O(n^2) memory usage, yet
  still retains much of the same information as FID.

  Args:
    activations1: 2D array of activations of size
      [num_images, num_dims] to use to compute Frechet Inception distance.
    activations2: 2D array of activations of size
      [num_images, num_dims] to use to compute Frechet Inception distance.

  Returns:
    The mean-only Frechet Inception distance. A floating-point scalar of the
    same type as the output of the activations.
  """
  activations1.shape.assert_has_rank(2)
  activations2.shape.assert_has_rank(2)

  activations_dtype = activations1.dtype
  if activations_dtype != tf.float64:
    activations1 = tf.cast(activations1, tf.float64)
    activations2 = tf.cast(activations2, tf.float64)

  # Compute means of activations.
  m = tf.reduce_mean(input_tensor=activations1, axis=0)
  m_w = tf.reduce_mean(input_tensor=activations2, axis=0)

  # Next the distance between means.
  mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(
      m, m_w))  # Equivalent to L2 but more stable.
  mofid = mean
  if activations_dtype != tf.float64:
    mofid = tf.cast(mofid, activations_dtype)

  return mofid


def diagonal_only_frechet_classifier_distance_from_activations(
    activations1, activations2):
  """Classifier distance for evaluating a generative model.

  This is based on the Frechet Inception distance, but for an arbitrary
  classifier.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calcuates

          |m - m_w|^2 + (sigma + sigma_w - 2(sigma x sigma_w)^(1/2))

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images. In this variant, we compute diagonal-only covariance matrices.
  As a result, instead of computing an expensive matrix square root, we can do
  something much simpler, and has O(n) vs O(n^2) space complexity.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute frechet classifier distance when comparing two
  generative models.

  Args:
    activations1: First activations to use to compute Frechet Inception
      distance.
    activations2: Second activations to use to compute Frechet Inception
      distance.

  Returns:
    The diagonal-only Frechet Inception distance. A floating-point scalar of
    the same type as the output of the activations.

  Raises:
    ValueError: If the shape of the variance and mean vectors are not equal.
  """
  activations1.shape.assert_has_rank(2)
  activations2.shape.assert_has_rank(2)

  activations_dtype = activations1.dtype
  if activations_dtype != tf.float64:
    activations1 = tf.cast(activations1, tf.float64)
    activations2 = tf.cast(activations2, tf.float64)

  # Compute mean and covariance matrices of activations.
  m, var = tf.nn.moments(x=activations1, axes=[0])
  m_w, var_w = tf.nn.moments(x=activations2, axes=[0])

  actual_shape = var.get_shape()
  expected_shape = m.get_shape()

  if actual_shape != expected_shape:
    raise ValueError('shape: {} must match expected shape: {}'.format(
        actual_shape, expected_shape))

  # Compute the two components of FID.

  # First the covariance component.
  # Here, note that trace(A + B) = trace(A) + trace(B)
  trace = tf.reduce_sum(
      input_tensor=(var + var_w) - 2.0 * tf.sqrt(tf.multiply(var, var_w)))

  # Next the distance between means.
  mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(
      m, m_w))  # Equivalent to L2 but more stable.
  dofid = trace + mean
  if activations_dtype != tf.float64:
    dofid = tf.cast(dofid, activations_dtype)

  return dofid


def _frechet_classifier_distance_from_activations_helper(
    activations1, activations2, streaming=False):
  """A helper function evaluating the frechet classifier distance."""
  activations1 = tf.convert_to_tensor(value=activations1)
  activations1.shape.assert_has_rank(2)
  activations2 = tf.convert_to_tensor(value=activations2)
  activations2.shape.assert_has_rank(2)

  activations_dtype = activations1.dtype
  if activations_dtype != tf.float64:
    activations1 = tf.cast(activations1, tf.float64)
    activations2 = tf.cast(activations2, tf.float64)

  # Compute mean and covariance matrices of activations.
  if streaming:
    m = eval_utils.streaming_mean_tensor_float64(
        tf.reduce_mean(input_tensor=activations1, axis=0))
    m_w = eval_utils.streaming_mean_tensor_float64(
        tf.reduce_mean(input_tensor=activations2, axis=0))
    sigma = eval_utils.streaming_covariance(activations1)
    sigma_w = eval_utils.streaming_covariance(activations2)
  else:
    m = (tf.reduce_mean(input_tensor=activations1, axis=0),)
    m_w = (tf.reduce_mean(input_tensor=activations2, axis=0),)
    # Calculate the unbiased covariance matrix of first activations.
    num_examples_real = tf.cast(tf.shape(input=activations1)[0], tf.float64)
    sigma = (num_examples_real / (num_examples_real - 1) *
             tfp.stats.covariance(activations1),)
    # Calculate the unbiased covariance matrix of second activations.
    num_examples_generated = tf.cast(
        tf.shape(input=activations2)[0], tf.float64)
    sigma_w = (num_examples_generated / (num_examples_generated - 1) *
               tfp.stats.covariance(activations2),)
  # m, m_w, sigma, sigma_w are tuples containing one or two elements: the first
  # element will be used to calculate the score value and the second will be
  # used to create the update_op. We apply the same operation on the two
  # elements to make sure their value is consistent.

  def _calculate_fid(m, m_w, sigma, sigma_w):
    """Returns the Frechet distance given the sample mean and covariance."""
    # Find the Tr(sqrt(sigma sigma_w)) component of FID
    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = tf.linalg.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    # Next the distance between means.
    mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(
        m, m_w))  # Equivalent to L2 but more stable.
    fid = trace + mean
    if activations_dtype != tf.float64:
      fid = tf.cast(fid, activations_dtype)
    return fid

  result = tuple(
      _calculate_fid(m_val, m_w_val, sigma_val, sigma_w_val)
      for m_val, m_w_val, sigma_val, sigma_w_val in zip(m, m_w, sigma, sigma_w))
  if streaming:
    return result
  else:
    return result[0]


def frechet_classifier_distance_from_activations(activations1, activations2):
  """Classifier distance for evaluating a generative model.

  This methods computes the Frechet classifier distance from activations of
  real images and generated images. This can be used independently of the
  frechet_classifier_distance() method, especially in the case of using large
  batches during evaluation where we would like precompute all of the
  activations before computing the classifier distance.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calculates

                |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute frechet classifier distance when comparing two
  generative models.

  Args:
    activations1: 2D Tensor containing activations. Shape is
      [batch_size, activation_size].
    activations2: 2D Tensor containing activations.
      [batch_size, activation_size].

  Returns:
   The Frechet Inception distance. A floating-point scalar of the same type
   as the output of the activations.
  """
  return _frechet_classifier_distance_from_activations_helper(
      activations1, activations2, streaming=False)


def frechet_classifier_distance_from_activations_streaming(
    activations1, activations2):
  """A streaming version of frechet_classifier_distance_from_activations.

  Keeps an internal state that continuously tracks the score. This internal
  state should be initialized with tf.initializers.local_variables().

  Args:
    activations1: 2D Tensor containing activations. Shape is
      [batch_size, activation_size].
    activations2: 2D Tensor containing activations. Shape is
      [batch_size, activation_size].

  Returns:
   A tuple containing the classifier score and a tf.Operation. The tf.Operation
   has the same value as the score, and has an additional side effect of
   updating the internal state with the given tensors.
  """
  return _frechet_classifier_distance_from_activations_helper(
      activations1, activations2, streaming=True)


def kernel_classifier_distance(input_tensor1,
                               input_tensor2,
                               classifier_fn,
                               num_batches=1,
                               max_block_size=1024,
                               dtype=None):
  """Kernel "classifier" distance for evaluating a generative model.

  This is based on the Kernel Inception distance, but for an arbitrary
  embedding.

  This technique is described in detail in https://arxiv.org/abs/1801.01401.
  Given two distributions P and Q of activations, this function calculates

      E_{X, X' ~ P}[k(X, X')] + E_{Y, Y' ~ Q}[k(Y, Y')]
        - 2 E_{X ~ P, Y ~ Q}[k(X, Y)]

  where k is the polynomial kernel

      k(x, y) = ( x^T y / dimension + 1 )^3.

  This captures how different the distributions of real and generated images'
  visual features are. Like the Frechet distance (and unlike the Inception
  score), this is a true distance and incorporates information about the
  target images. Unlike the Frechet score, this function computes an
  *unbiased* and asymptotically normal estimator, which makes comparing
  estimates across models much more intuitive.

  The estimator used takes time quadratic in max_block_size. Larger values of
  max_block_size will decrease the variance of the estimator but increase the
  computational cost. This differs slightly from the estimator used by the
  original paper; it is the block estimator of https://arxiv.org/abs/1307.1954.

  NOTE: the blocking code assumes that real_activations and
  generated_activations are both in random order. If either is sorted in a
  meaningful order, the estimator will behave poorly.

  NOTE: This function consumes images, computes their activations, and then
  computes the classifier score. If you would like to precompute many
  activations for real and generated images for large batches, or to compute
  multiple scores based on the same images, please use
  kernel_clasifier_distance_from_activations(), which this method also uses.

  Args:
    input_tensor1: First input to use to compute Kernel Inception distance.
    input_tensor2: Second input to use to compute Kernel Inception distance.
    classifier_fn: A function that takes tensors and produces activations based
      on a classifier.
    num_batches: Number of batches to split images in to in order to
      efficiently run them through the classifier network.
    max_block_size: integer, default 1024. The distance estimator splits samples
      into blocks for computational efficiency. Larger values are more
      computationally expensive but decrease the variance of the distance
      estimate.
    dtype: if not None, coerce activations to this dtype before computations.

  Returns:
   The Kernel Inception Distance. A floating-point scalar of the same type
   as the output of the activations.
  """
  return kernel_classifier_distance_and_std(
      input_tensor1,
      input_tensor2,
      classifier_fn,
      num_batches=num_batches,
      max_block_size=max_block_size,
      dtype=dtype)[0]


def kernel_classifier_distance_and_std(input_tensor1,
                                       input_tensor2,
                                       classifier_fn,
                                       num_batches=1,
                                       max_block_size=1024,
                                       dtype=None):
  """Kernel "classifier" distance for evaluating a generative model.

  This is based on the Kernel Inception distance, but for an arbitrary
  embedding. Also returns an estimate of the standard error of the distance
  estimator.

  This technique is described in detail in https://arxiv.org/abs/1801.01401.
  Given two distributions P and Q of activations, this function calculates

      E_{X, X' ~ P}[k(X, X')] + E_{Y, Y' ~ Q}[k(Y, Y')]
        - 2 E_{X ~ P, Y ~ Q}[k(X, Y)]

  where k is the polynomial kernel

      k(x, y) = ( x^T y / dimension + 1 )^3.

  This captures how different the distributions of real and generated images'
  visual features are. Like the Frechet distance (and unlike the Inception
  score), this is a true distance and incorporates information about the
  target images. Unlike the Frechet score, this function computes an
  *unbiased* and asymptotically normal estimator, which makes comparing
  estimates across models much more intuitive.

  The estimator used takes time quadratic in max_block_size. Larger values of
  max_block_size will decrease the variance of the estimator but increase the
  computational cost. This differs slightly from the estimator used by the
  original paper; it is the block estimator of https://arxiv.org/abs/1307.1954.

  NOTE: the blocking code assumes that real_activations and
  generated_activations are both in random order. If either is sorted in a
  meaningful order, the estimator will behave poorly.

  NOTE: This function consumes images, computes their activations, and then
  computes the classifier score. If you would like to precompute many
  activations for real and generated images for large batches, or to compute
  multiple scores based on the same images, please use
  kernel_clasifier_distance_from_activations(), which this method also uses.

  Args:
    input_tensor1: Input tensor to use to compute Kernel Inception distance.
    input_tensor2: Input tensor to use to compute Kernel Inception distance.
    classifier_fn: A function that takes tensors and produces activations based
      on a classifier.
    num_batches: Number of batches to split images in to in order to
      efficiently run them through the classifier network.
    max_block_size: integer, default 1024. The distance estimator splits samples
      into blocks for computational efficiency. Larger values are more
      computationally expensive but decrease the variance of the distance
      estimate. Having a smaller block size also gives a better estimate of the
      standard error.
    dtype: if not None, coerce activations to this dtype before computations.

  Returns:
   The Kernel Inception Distance. A floating-point scalar of the same type
     as the output of the activations.
   An estimate of the standard error of the distance estimator (a scalar of
     the same type).
  """
  input_list1 = tf.split(input_tensor1, num_or_size_splits=num_batches)
  input_list2 = tf.split(input_tensor2, num_or_size_splits=num_batches)

  stack1 = tf.stack(input_list1)
  stack2 = tf.stack(input_list2)

  # Compute the activations using the memory-efficient `map_fn`.
  def compute_activations(elems):
    return tf.map_fn(
        fn=classifier_fn,
        elems=elems,
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')

  acts1 = compute_activations(stack1)
  acts2 = compute_activations(stack2)

  # Ensure the activations have the right shapes.
  acts1 = tf.concat(tf.unstack(acts1), 0)
  acts2 = tf.concat(tf.unstack(acts2), 0)

  return kernel_classifier_distance_and_std_from_activations(
      acts1, acts2, max_block_size, dtype)


def kernel_classifier_distance_from_activations(activations1,
                                                activations2,
                                                max_block_size=1024,
                                                dtype=None):
  """Kernel "classifier" distance for evaluating a generative model.

  This methods computes the kernel classifier distance from activations of
  real images and generated images. This can be used independently of the
  kernel_classifier_distance() method, especially in the case of using large
  batches during evaluation where we would like to precompute all of the
  activations before computing the classifier distance, or if we want to
  compute multiple metrics based on the same images.

  This technique is described in detail in https://arxiv.org/abs/1801.01401.
  Given two distributions P and Q of activations, this function calculates

      E_{X, X' ~ P}[k(X, X')] + E_{Y, Y' ~ Q}[k(Y, Y')]
        - 2 E_{X ~ P, Y ~ Q}[k(X, Y)]

  where k is the polynomial kernel

      k(x, y) = ( x^T y / dimension + 1 )^3.

  This captures how different the distributions of real and generated images'
  visual features are. Like the Frechet distance (and unlike the Inception
  score), this is a true distance and incorporates information about the
  target images. Unlike the Frechet score, this function computes an
  *unbiased* and asymptotically normal estimator, which makes comparing
  estimates across models much more intuitive.

  The estimator used takes time quadratic in max_block_size. Larger values of
  max_block_size will decrease the variance of the estimator but increase the
  computational cost. This differs slightly from the estimator used by the
  original paper; it is the block estimator of https://arxiv.org/abs/1307.1954.

  NOTE: the blocking code assumes that real_activations and
  generated_activations are both in random order. If either is sorted in a
  meaningful order, the estimator will behave poorly.

  Args:
    activations1: 2D Tensor containing activations. Shape is
      [batch_size, activation_size].
    activations2: 2D Tensor containing activations. Shape is
      [batch_size, activation_size].
    max_block_size: integer, default 1024. The distance estimator splits samples
      into blocks for computational efficiency. Larger values are more
      computationally expensive but decrease the variance of the distance
      estimate.
    dtype: If not None, coerce activations to this dtype before computations.

  Returns:
   The Kernel Inception Distance. A floating-point scalar of the same type
   as the output of the activations.
  """
  return kernel_classifier_distance_and_std_from_activations(
      activations1, activations2, max_block_size, dtype)[0]


def kernel_classifier_distance_and_std_from_activations(activations1,
                                                        activations2,
                                                        max_block_size=1024,
                                                        dtype=None):
  """Kernel "classifier" distance for evaluating a generative model.

  This methods computes the kernel classifier distance from activations of
  real images and generated images. This can be used independently of the
  kernel_classifier_distance() method, especially in the case of using large
  batches during evaluation where we would like to precompute all of the
  activations before computing the classifier distance, or if we want to
  compute multiple metrics based on the same images. It also returns a rough
  estimate of the standard error of the estimator.

  This technique is described in detail in https://arxiv.org/abs/1801.01401.
  Given two distributions P and Q of activations, this function calculates

      E_{X, X' ~ P}[k(X, X')] + E_{Y, Y' ~ Q}[k(Y, Y')]
        - 2 E_{X ~ P, Y ~ Q}[k(X, Y)]

  where k is the polynomial kernel

      k(x, y) = ( x^T y / dimension + 1 )^3.

  This captures how different the distributions of real and generated images'
  visual features are. Like the Frechet distance (and unlike the Inception
  score), this is a true distance and incorporates information about the
  target images. Unlike the Frechet score, this function computes an
  *unbiased* and asymptotically normal estimator, which makes comparing
  estimates across models much more intuitive.

  The estimator used takes time quadratic in max_block_size. Larger values of
  max_block_size will decrease the variance of the estimator but increase the
  computational cost. This differs slightly from the estimator used by the
  original paper; it is the block estimator of https://arxiv.org/abs/1307.1954.
  The estimate of the standard error will also be more reliable when there are
  more blocks, i.e. when max_block_size is smaller.

  NOTE: the blocking code assumes that real_activations and
  generated_activations are both in random order. If either is sorted in a
  meaningful order, the estimator will behave poorly.

  Args:
    activations1: 2D Tensor containing activations. Shape is
      [batch_size, activation_size].
    activations2: 2D Tensor containing activations. Shape is
      [batch_size, activation_size].
    max_block_size: integer, default 1024. The distance estimator splits samples
      into blocks for computational efficiency. Larger values are more
      computationally expensive but decrease the variance of the distance
      estimate. Having a smaller block size also gives a better estimate of the
      standard error.
    dtype: If not None, coerce activations to this dtype before computations.

  Returns:
   The Kernel Inception Distance. A floating-point scalar of the same type
     as the output of the activations.
   An estimate of the standard error of the distance estimator (a scalar of
     the same type).
  """
  activations1.shape.assert_has_rank(2)
  activations2.shape.assert_has_rank(2)
  activations1.shape[1:2].assert_is_compatible_with(activations2.shape[1:2])

  if dtype is None:
    dtype = activations1.dtype
    assert activations2.dtype == dtype
  else:
    activations1 = tf.cast(activations1, dtype)
    activations2 = tf.cast(activations2, dtype)

  # Figure out how to split the activations into blocks of approximately
  # equal size, with none larger than max_block_size.
  n_r = tf.shape(input=activations1)[0]
  n_g = tf.shape(input=activations2)[0]

  n_bigger = tf.maximum(n_r, n_g)
  n_blocks = tf.cast(tf.math.ceil(n_bigger / max_block_size), dtype=tf.int32)

  v_r = n_r // n_blocks
  v_g = n_g // n_blocks

  n_plusone_r = n_r - v_r * n_blocks
  n_plusone_g = n_g - v_g * n_blocks

  sizes_r = tf.concat([
      tf.fill([n_blocks - n_plusone_r], v_r),
      tf.fill([n_plusone_r], v_r + 1),
  ], 0)
  sizes_g = tf.concat([
      tf.fill([n_blocks - n_plusone_g], v_g),
      tf.fill([n_plusone_g], v_g + 1),
  ], 0)

  zero = tf.zeros([1], dtype=tf.int32)
  inds_r = tf.concat([zero, tf.cumsum(sizes_r)], 0)
  inds_g = tf.concat([zero, tf.cumsum(sizes_g)], 0)

  dim = tf.cast(activations1.shape[1], dtype)

  def compute_kid_block(i):
    """Computes the ith block of the KID estimate."""
    r_s = inds_r[i]
    r_e = inds_r[i + 1]
    r = activations1[r_s:r_e]
    m = tf.cast(r_e - r_s, dtype)

    g_s = inds_g[i]
    g_e = inds_g[i + 1]
    g = activations2[g_s:g_e]
    n = tf.cast(g_e - g_s, dtype)

    k_rr = (tf.matmul(r, r, transpose_b=True) / dim + 1)**3
    k_rg = (tf.matmul(r, g, transpose_b=True) / dim + 1)**3
    k_gg = (tf.matmul(g, g, transpose_b=True) / dim + 1)**3
    return (-2 * tf.reduce_mean(input_tensor=k_rg) +
            (tf.reduce_sum(input_tensor=k_rr) - tf.linalg.trace(k_rr)) /
            (m * (m - 1)) +
            (tf.reduce_sum(input_tensor=k_gg) - tf.linalg.trace(k_gg)) /
            (n * (n - 1)))

  ests = tf.map_fn(
      compute_kid_block, tf.range(n_blocks), dtype=dtype, back_prop=False)

  mn = tf.reduce_mean(input_tensor=ests)

  # tf.nn.moments doesn't use the Bessel correction, which we want here
  n_blocks_ = tf.cast(n_blocks, dtype)
  var = tf.cond(
      pred=tf.less_equal(n_blocks, 1),
      true_fn=lambda: tf.constant(float('nan'), dtype=dtype),
      false_fn=lambda: tf.reduce_sum(input_tensor=tf.square(ests - mn)) / (
          n_blocks_ - 1))

  return mn, tf.sqrt(var / n_blocks_)
