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

"""Model evaluation tools for TF-GAN.

These methods come from https://arxiv.org/abs/1606.03498,
https://arxiv.org/abs/1706.08500, and https://arxiv.org/abs/1801.01401.

NOTE: This implementation uses the same weights as in
https://github.com/openai/improved-gan/blob/master/inception_score/model.py,
but is more numerically stable and is an unbiased estimator of the true
Inception score even when splitting the inputs into batches.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys
import tarfile

import six
from six.moves import urllib

import tensorflow as tf

__all__ = [
    'get_graph_def_from_disk',
    'get_graph_def_from_resource',
    'get_graph_def_from_url_tarball',
    'preprocess_image',
    'run_image_classifier',
    'sample_and_run_image_classifier',
    'sample_and_run_classifier_fn',
    'run_inception',
    'sample_and_run_inception',
    'inception_score',
    'classifier_score',
    'classifier_score_from_logits',
    'frechet_inception_distance',
    'frechet_classifier_distance',
    'frechet_classifier_distance_from_activations',
    'mean_only_frechet_classifier_distance_from_activations',
    'diagonal_only_frechet_classifier_distance_from_activations',
    'kernel_inception_distance',
    'kernel_inception_distance_and_std',
    'kernel_classifier_distance',
    'kernel_classifier_distance_and_std',
    'kernel_classifier_distance_from_activations',
    'kernel_classifier_distance_and_std_from_activations',
    'INCEPTION_DEFAULT_IMAGE_SIZE',
    'INCEPTION_OUTPUT',
    'INCEPTION_FINAL_POOL',
]

INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'
INCEPTION_INPUT = 'Mul:0'
INCEPTION_OUTPUT = 'logits:0'
INCEPTION_FINAL_POOL = 'pool_3:0'
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def _validate_images(images, image_size):
  images = tf.convert_to_tensor(value=images)
  images.shape.with_rank(4)
  images.shape.assert_is_compatible_with([None, image_size, image_size, None])
  return images


def _to_float(tensor):
  return tf.cast(tensor, tf.float32)


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
  si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
  # Note that the v returned by Tensorflow is v = V
  # (when referencing the equation A = U S V^T)
  # This is unlike Numpy which returns v = V^T
  return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)


def preprocess_image(images,
                     height=INCEPTION_DEFAULT_IMAGE_SIZE,
                     width=INCEPTION_DEFAULT_IMAGE_SIZE,
                     scope=None):
  """Prepare a batch of images for evaluation.

  This is the preprocessing portion of the graph from
  http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz.

  Note that it expects Tensors in [0, 255]. This function maps pixel values to
  [-1, 1] and resizes to match the InceptionV1 network.

  Args:
    images: 3-D or 4-D Tensor of images. Values are in [0, 255].
    height: Integer. Height of resized output image.
    width: Integer. Width of resized output image.
    scope: Optional scope for name_scope.

  Returns:
    3-D or 4-D float Tensor of prepared image(s). Values are in [-1, 1].
  """
  is_single = images.shape.ndims == 3
  with tf.compat.v1.name_scope(scope, 'preprocess', [images, height, width]):
    if not images.dtype.is_floating:
      images = _to_float(images)
    if is_single:
      images = tf.expand_dims(images, axis=0)
    resized = tf.compat.v1.image.resize(
        images, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    resized = (resized - 128.0) / 128.0
    if is_single:
      resized = tf.squeeze(resized, axis=0)
    return resized


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


def get_graph_def_from_disk(filename):
  """Get a GraphDef proto from a disk location."""
  with tf.compat.v1.gfile.GFile(filename, 'rb') as f:
    return tf.compat.v1.GraphDef.FromString(f.read())


def get_graph_def_from_resource(filename):
  """Get a GraphDef proto from within a .par file."""
  return tf.compat.v1.GraphDef.FromString(
      tf.compat.v1.resource_loader.load_resource(filename))


def get_graph_def_from_url_tarball(url, filename, tar_filename=None):
  """Get a GraphDef proto from a tarball on the web.

  Args:
    url: Web address of tarball
    filename: Filename of graph definition within tarball
    tar_filename: Temporary download filename (None = always download)

  Returns:
    A GraphDef loaded from a file in the downloaded tarball.
  """
  if not (tar_filename and os.path.exists(tar_filename)):

    def _progress(count, block_size, total_size):
      sys.stdout.write(
          '\r>> Downloading %s %.1f%%' %
          (url, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    tar_filename, _ = urllib.request.urlretrieve(url, tar_filename, _progress)
  with tarfile.open(tar_filename, 'r:gz') as tar:
    proto_str = tar.extractfile(filename).read()
  return tf.compat.v1.GraphDef.FromString(proto_str)


def _default_graph_def_fn():
  return get_graph_def_from_url_tarball(INCEPTION_URL, INCEPTION_FROZEN_GRAPH,
                                        os.path.basename(INCEPTION_URL))


def _graph_def_or_default(graph_def, default_graph_def_fn):
  if graph_def is None:
    if default_graph_def_fn is None:
      raise ValueError('If `graph_def` is `None`, must provide '
                       '`default_graph_def_fn`.')
    return default_graph_def_fn()
  return graph_def


def _flatten_activations_or_list(activations):
  if isinstance(activations, list):
    for i, activation in enumerate(activations):
      if tf.rank(activation) != 2:
        activations[i] = tf.compat.v1.layers.flatten(activation)
  else:
    if tf.rank(activations) != 2:
      activations = tf.compat.v1.layers.flatten(activations)
  return activations


def run_inception(images,
                  graph_def=None,
                  default_graph_def_fn=_default_graph_def_fn,
                  image_size=INCEPTION_DEFAULT_IMAGE_SIZE,
                  input_tensor=INCEPTION_INPUT,
                  output_tensor=INCEPTION_OUTPUT,
                  num_batches=1):
  """Run images through a pretrained Inception classifier.

  Args:
    images: Input tensors. Must be [batch, height, width, channels]. Input shape
      and values must be in [-1, 1], which can be achieved using
      `preprocess_image`.
    graph_def: A GraphDef proto of a pretrained Inception graph. If `None`, call
      `default_graph_def_fn` to get GraphDef.
    default_graph_def_fn: A function that returns a GraphDef. Used if
      `graph_def` is `None. By default, returns a pretrained InceptionV3 graph.
    image_size: Required image width and height. See unit tests for the default
      values.
    input_tensor: Name of input Tensor.
    output_tensor: Name or list of output Tensors. This function will compute
      activations at the specified layer. Examples include INCEPTION_V3_OUTPUT
      and INCEPTION_V3_FINAL_POOL which would result in this function computing
      the final logits or the penultimate pooling layer.
    num_batches: Number of batches to split `images` in to in order to
      efficiently run them through the classifier network.

  Returns:
    Tensor or Tensors corresponding to computed `output_tensor`.

  Raises:
    ValueError: If images are not the correct size.
    ValueError: If neither `graph_def` nor `default_graph_def_fn` are provided.
  """
  images = _validate_images(images, image_size)
  graph_def = _graph_def_or_default(graph_def, default_graph_def_fn)
  activations = run_image_classifier(images, graph_def, input_tensor,
                                     output_tensor, num_batches)
  return _flatten_activations_or_list(activations)


def sample_and_run_inception(sample_fn,
                             sample_inputs,
                             graph_def=None,
                             default_graph_def_fn=_default_graph_def_fn,
                             input_tensor=INCEPTION_INPUT,
                             output_tensor=INCEPTION_OUTPUT):
  """Generates images then passes them through an Inception classifier.

  This is the same as `run_inception`, but instead of taking images it takes a
  function that samples from an image distribution. This is essential when
  running inception on large batches, since it may be impossible to hold all the
  images in memory at once.

  NOTE: Running the sampler can affect the original weights if, for instance,
  there are assign ops in the sampler. See
  `test_assign_variables_in_sampler_runs` in the unit tests for an example.

  Args:
    sample_fn: A function that takes a single argument and returns images. This
      function samples from an image distribution.
    sample_inputs: A list of inputs to pass to `sample_fn`.
    graph_def: A GraphDef proto of a pretrained Inception graph. If `None`, call
      `default_graph_def_fn` to get GraphDef.
    default_graph_def_fn: A function that returns a GraphDef. Used if
      `graph_def` is `None. By default, returns a pretrained InceptionV3 graph.
    input_tensor: Name of input Tensor.
    output_tensor: Name or list of output Tensors. This function will compute
      activations at the specified layer. Examples include INCEPTION_V3_OUTPUT
      and INCEPTION_V3_FINAL_POOL which would result in this function computing
      the final logits or the penultimate pooling layer.

  Returns:
    Tensor or Tensors corresponding to computed `output_tensor`.

  Raises:
    ValueError: If images are not the correct size.
    ValueError: If neither `graph_def` nor `default_graph_def_fn` are provided.
  """
  graph_def = _graph_def_or_default(graph_def, default_graph_def_fn)
  activations = sample_and_run_image_classifier(
      sample_fn, sample_inputs, graph_def, input_tensor, output_tensor)
  return _flatten_activations_or_list(activations)


def _combine_outputs_after_while_loop(classifier_outputs):
  if isinstance(classifier_outputs, tf.Tensor):
    classifier_outputs = tf.concat(tf.unstack(classifier_outputs), 0)
  else:
    classifier_outputs = [tf.concat(tf.unstack(x), 0) for x in
                          classifier_outputs]
  return classifier_outputs


def run_image_classifier(tensor,
                         graph_def,
                         input_tensor,
                         output_tensor,
                         num_batches=1,
                         dtypes=None,
                         scope='RunClassifier'):
  """Runs a network from a frozen graph.

  If there are multiple outputs, cast them to tf.float32.

  Args:
    tensor: An Input tensor.
    graph_def: A GraphDef proto.
    input_tensor: Name of input tensor in graph def.
    output_tensor: A tensor name or list of tensor names in graph def.
    num_batches: Number of batches to split `tensor` in to in order to
      efficiently run them through the classifier network. This is useful if
      running a large batch would consume too much memory, but running smaller
      batches is feasible.
    dtypes: If `output_tensor` is a list, the `while_loop` must have a dtype for
      every output. If `dtypes` is `None` in this case, assume every output type
      is `tf.float32`.
    scope: Name scope for classifier.

  Returns:
    Classifier output if `output_tensor` is a string, or a list of outputs if
    `output_tensor` is a list.

  Raises:
    ValueError: If `input_tensor` or `output_tensor` aren't in the graph_def.
    ValueError: If executing eagerly.
  """
  if tf.executing_eagerly():
    raise ValueError('`run_image_classifier` doesn\'t work in eager.')

  if isinstance(output_tensor, six.string_types):
    output_tensor = [output_tensor]
    output_is_tensor = True
  elif len(output_tensor) == 1:
    output_is_tensor = False
  else:
    output_is_tensor = False
    dtypes = dtypes or [tf.float32] * len(output_tensor)

  def _fn(tensor):
    input_map = {input_tensor: tensor}
    outputs = tf.graph_util.import_graph_def(
        graph_def, input_map, output_tensor, name=scope)
    if len(outputs) == 1:
      outputs = outputs[0]
    return outputs
  if num_batches > 1:
    # Compute the classifier splits using the memory-efficient `map_fn`.
    input_list = tf.split(tensor, num_or_size_splits=num_batches)
    classifier_outputs = tf.map_fn(
        fn=_fn,
        elems=tf.stack(input_list),
        dtype=dtypes,
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    classifier_outputs = _combine_outputs_after_while_loop(classifier_outputs)
  else:
    classifier_outputs = _fn(tensor)
  if not output_is_tensor and not isinstance(classifier_outputs, list):
    classifier_outputs = [classifier_outputs]

  return classifier_outputs


def sample_and_run_image_classifier(sample_fn,
                                    sample_inputs,
                                    graph_def,
                                    input_tensor,
                                    output_tensor,
                                    dtypes=None,
                                    scope='SampleAndRunClassifier'):
  """Sampes Tensors from distribution then runs them through a frozen graph.

  This is the same as `run_image_classifier`, but instead of taking images it
  takes a function that samples from an image distribution. This is essential
  when running inception on large batches, since it may be impossible to hold
  all the images in memory at once.

  If there are multiple outputs, cast them to tf.float32.

  NOTE: Running the sampler can affect the original weights if, for instance,
  there are assign ops in the sampler. See
  `test_assign_variables_in_sampler_runs` in the unit tests for an example.

  Args:
    sample_fn: A function that takes a single argument and returns images. This
      function samples from an image distribution.
    sample_inputs: A list of inputs to pass to `sample_fn`.
    graph_def: A GraphDef proto.
    input_tensor: Name of input tensor in graph def.
    output_tensor: A tensor name or list of tensor names in graph def.
    dtypes: If `output_tensor` is a list, the `while_loop` must have a dtype for
      every output. If `dtypes` is `None` in this case, assume every output type
      is `tf.float32`.
    scope: Name scope for classifier.

  Returns:
    Classifier output if `output_tensor` is a string, or a list of outputs if
    `output_tensor` is a list.

  Raises:
    ValueError: If `input_tensor` or `output_tensor` aren't in the graph_def.
  """
  if isinstance(output_tensor, six.string_types):
    dtypes = dtypes or tf.float32
  else:
    dtypes = dtypes or [tf.float32] * len(output_tensor)

  def _classifier_fn(tensor):
    return run_image_classifier(
        tensor, graph_def, input_tensor, output_tensor)

  classifier_outputs = sample_and_run_classifier_fn(
      sample_fn, sample_inputs, _classifier_fn, dtypes, scope)

  if (isinstance(output_tensor, list) and
      not isinstance(classifier_outputs, list)):
    classifier_outputs = [classifier_outputs]

  return classifier_outputs


def sample_and_run_classifier_fn(sample_fn,
                                 sample_inputs,
                                 classifier_fn,
                                 dtypes=None,
                                 scope='SampleAndRunClassifierFn'):
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
      outputs of the classifier.
    dtypes: If `output_tensor` is a list, the `while_loop` must have a dtype for
      every output. If `dtypes` is `None` in this case, assume every output type
      is `tf.float32`.
    scope: Name scope for classifier.

  Returns:
    Classifier output if `output_tensor` is a string, or a list of outputs if
    `output_tensor` is a list.

  Raises:
    ValueError: If `input_tensor` or `output_tensor` aren't in the graph_def.
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
        name=scope)
    classifier_outputs = _combine_outputs_after_while_loop(classifier_outputs)
  else:
    classifier_outputs = _fn(sample_inputs[0])

  return classifier_outputs


def classifier_score(images, classifier_fn, num_batches=1):
  """Classifier score for evaluating a conditional generative model.

  This is based on the Inception Score, but for an arbitrary classifier.

  This technique is described in detail in https://arxiv.org/abs/1606.03498. In
  summary, this function calculates

  exp( E[ KL(p(y|x) || p(y)) ] )

  which captures how different the network's classification prediction is from
  the prior distribution over classes.

  NOTE: This function consumes images, computes their logits, and then
  computes the classifier score. If you would like to precompute many logits for
  large batches, use classifier_score_from_logits(), which this method also
  uses.

  Args:
    images: Images to calculate the classifier score for.
    classifier_fn: A function that takes images and produces logits based on a
      classifier.
    num_batches: Number of batches to split `generated_images` in to in order to
      efficiently run them through the classifier network.

  Returns:
    The classifier score. A floating-point scalar of the same type as the output
    of `classifier_fn`.
  """
  generated_images_list = tf.split(images, num_or_size_splits=num_batches)

  # Compute the classifier splits using the memory-efficient `map_fn`.
  logits = tf.map_fn(
      fn=classifier_fn,
      elems=tf.stack(generated_images_list),
      parallel_iterations=1,
      back_prop=False,
      swap_memory=True,
      name='RunClassifier')
  logits = tf.concat(tf.unstack(logits), 0)

  return classifier_score_from_logits(logits)


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
  logits = tf.convert_to_tensor(value=logits)
  logits.shape.assert_has_rank(2)

  # Use maximum precision for best results.
  logits_dtype = logits.dtype
  if logits_dtype != tf.float64:
    logits = tf.cast(logits, tf.float64)

  p = tf.nn.softmax(logits)
  q = tf.reduce_mean(input_tensor=p, axis=0)
  kl = kl_divergence(p, logits, q)
  kl.shape.assert_has_rank(1)
  log_score = tf.reduce_mean(input_tensor=kl)
  final_score = tf.exp(log_score)

  if logits_dtype != tf.float64:
    final_score = tf.cast(final_score, logits_dtype)

  return final_score


inception_score = functools.partial(
    classifier_score,
    classifier_fn=functools.partial(
        run_inception, output_tensor=INCEPTION_OUTPUT))


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


def frechet_classifier_distance(real_images,
                                generated_images,
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

  NOTE: This function consumes images, computes their activations, and then
  computes the classifier score. If you would like to precompute many
  activations for real and generated images for large batches, please use
  frechet_clasifier_distance_from_activations(), which this method also uses.

  Args:
    real_images: Real images to use to compute Frechet Inception distance.
    generated_images: Generated images to use to compute Frechet Inception
      distance.
    classifier_fn: A function that takes images and produces activations based
      on a classifier.
    num_batches: Number of batches to split images in to in order to efficiently
      run them through the classifier network.

  Returns:
    The Frechet Inception distance. A floating-point scalar of the same type
    as the output of `classifier_fn`.
  """
  real_images_list = tf.split(real_images, num_or_size_splits=num_batches)
  generated_images_list = tf.split(
      generated_images, num_or_size_splits=num_batches)

  real_imgs = tf.stack(real_images_list)
  generated_imgs = tf.stack(generated_images_list)

  # Compute the activations using the memory-efficient `map_fn`.
  def compute_activations(elems):
    return tf.map_fn(
        fn=classifier_fn,
        elems=elems,
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')

  real_a = compute_activations(real_imgs)
  gen_a = compute_activations(generated_imgs)

  # Ensure the activations have the right shapes.
  real_a = tf.concat(tf.unstack(real_a), 0)
  gen_a = tf.concat(tf.unstack(gen_a), 0)

  return frechet_classifier_distance_from_activations(real_a, gen_a)


def mean_only_frechet_classifier_distance_from_activations(
    real_activations, generated_activations):
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
    real_activations: 2D array of activations of real images of size
      [num_images, num_dims] to use to compute Frechet Inception distance.
    generated_activations: 2D array of activations of generated images of size
      [num_images, num_dims] to use to compute Frechet Inception distance.

  Returns:
    The mean-only Frechet Inception distance. A floating-point scalar of the
    same type as the output of the activations.
  """
  real_activations.shape.assert_has_rank(2)
  generated_activations.shape.assert_has_rank(2)

  activations_dtype = real_activations.dtype
  if activations_dtype != tf.float64:
    real_activations = tf.cast(real_activations, tf.float64)
    generated_activations = tf.cast(generated_activations, tf.float64)

  # Compute means of activations.
  m = tf.reduce_mean(input_tensor=real_activations, axis=0)
  m_w = tf.reduce_mean(input_tensor=generated_activations, axis=0)

  # Next the distance between means.
  mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(
      m, m_w))  # Equivalent to L2 but more stable.
  mofid = mean
  if activations_dtype != tf.float64:
    mofid = tf.cast(mofid, activations_dtype)

  return mofid


def diagonal_only_frechet_classifier_distance_from_activations(
    real_activations, generated_activations):
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
    real_activations: Real images to use to compute Frechet Inception distance.
    generated_activations: Generated images to use to compute Frechet Inception
      distance.

  Returns:
    The diagonal-only Frechet Inception distance. A floating-point scalar of
    the same type as the output of the activations.

  Raises:
    ValueError: If the shape of the variance and mean vectors are not equal.
  """
  real_activations.shape.assert_has_rank(2)
  generated_activations.shape.assert_has_rank(2)

  activations_dtype = real_activations.dtype
  if activations_dtype != tf.float64:
    real_activations = tf.cast(real_activations, tf.float64)
    generated_activations = tf.cast(generated_activations, tf.float64)

  # Compute mean and covariance matrices of activations.
  m, var = tf.nn.moments(x=real_activations, axes=[0])
  m_w, var_w = tf.nn.moments(x=generated_activations, axes=[0])

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


def frechet_classifier_distance_from_activations(real_activations,
                                                 generated_activations):
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
    real_activations: 2D Tensor containing activations of real data. Shape is
      [batch_size, activation_size].
    generated_activations: 2D Tensor containing activations of generated data.
      Shape is [batch_size, activation_size].

  Returns:
   The Frechet Inception distance. A floating-point scalar of the same type
   as the output of the activations.

  """
  real_activations = tf.convert_to_tensor(value=real_activations)
  real_activations.shape.assert_has_rank(2)
  generated_activations = tf.convert_to_tensor(value=generated_activations)
  generated_activations.shape.assert_has_rank(2)

  activations_dtype = real_activations.dtype
  if activations_dtype != tf.float64:
    real_activations = tf.cast(real_activations, tf.float64)
    generated_activations = tf.cast(generated_activations, tf.float64)

  # Compute mean and covariance matrices of activations.
  m = tf.reduce_mean(input_tensor=real_activations, axis=0)
  m_w = tf.reduce_mean(input_tensor=generated_activations, axis=0)
  num_examples_real = tf.cast(tf.shape(input=real_activations)[0], tf.float64)
  num_examples_generated = tf.cast(
      tf.shape(input=generated_activations)[0], tf.float64)

  # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
  real_centered = real_activations - m
  sigma = tf.matmul(
      real_centered, real_centered, transpose_a=True) / (
          num_examples_real - 1)

  gen_centered = generated_activations - m_w
  sigma_w = tf.matmul(
      gen_centered, gen_centered, transpose_a=True) / (
          num_examples_generated - 1)

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


frechet_inception_distance = functools.partial(
    frechet_classifier_distance,
    classifier_fn=functools.partial(
        run_inception, output_tensor=INCEPTION_FINAL_POOL))


def kernel_classifier_distance(real_images,
                               generated_images,
                               classifier_fn,
                               num_classifier_batches=1,
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
    real_images: Real images to use to compute Kernel Inception distance.
    generated_images: Generated images to use to compute Kernel Inception
      distance.
    classifier_fn: A function that takes images and produces activations based
      on a classifier.
    num_classifier_batches: Number of batches to split images in to in order to
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
      real_images,
      generated_images,
      classifier_fn,
      num_classifier_batches=num_classifier_batches,
      max_block_size=max_block_size,
      dtype=dtype)[0]


kernel_inception_distance = functools.partial(
    kernel_classifier_distance,
    classifier_fn=functools.partial(
        run_inception, output_tensor=INCEPTION_FINAL_POOL))


def kernel_classifier_distance_and_std(real_images,
                                       generated_images,
                                       classifier_fn,
                                       num_classifier_batches=1,
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
    real_images: Real images to use to compute Kernel Inception distance.
    generated_images: Generated images to use to compute Kernel Inception
      distance.
    classifier_fn: A function that takes images and produces activations based
      on a classifier.
    num_classifier_batches: Number of batches to split images in to in order to
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
  real_images_list = tf.split(
      real_images, num_or_size_splits=num_classifier_batches)
  generated_images_list = tf.split(
      generated_images, num_or_size_splits=num_classifier_batches)

  real_imgs = tf.stack(real_images_list)
  generated_imgs = tf.stack(generated_images_list)

  # Compute the activations using the memory-efficient `map_fn`.
  def compute_activations(elems):
    return tf.map_fn(
        fn=classifier_fn,
        elems=elems,
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')

  real_a = compute_activations(real_imgs)
  gen_a = compute_activations(generated_imgs)

  # Ensure the activations have the right shapes.
  real_a = tf.concat(tf.unstack(real_a), 0)
  gen_a = tf.concat(tf.unstack(gen_a), 0)

  return kernel_classifier_distance_and_std_from_activations(
      real_a, gen_a, max_block_size, dtype)


kernel_inception_distance_and_std = functools.partial(
    kernel_classifier_distance_and_std,
    classifier_fn=functools.partial(
        run_inception, output_tensor=INCEPTION_FINAL_POOL))


def kernel_classifier_distance_from_activations(real_activations,
                                                generated_activations,
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
    real_activations: 2D Tensor containing activations of real data. Shape is
      [batch_size, activation_size].
    generated_activations: 2D Tensor containing activations of generated data.
      Shape is [batch_size, activation_size].
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
      real_activations, generated_activations, max_block_size, dtype)[0]


def kernel_classifier_distance_and_std_from_activations(real_activations,
                                                        generated_activations,
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
    real_activations: 2D Tensor containing activations of real data. Shape is
      [batch_size, activation_size].
    generated_activations: 2D Tensor containing activations of generated data.
      Shape is [batch_size, activation_size].
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

  real_activations.shape.assert_has_rank(2)
  generated_activations.shape.assert_has_rank(2)
  real_activations.shape[1:2].assert_is_compatible_with(
      generated_activations.shape[1:2])

  if dtype is None:
    dtype = real_activations.dtype
    assert generated_activations.dtype == dtype
  else:
    real_activations = tf.cast(real_activations, dtype)
    generated_activations = tf.cast(generated_activations, dtype)

  # Figure out how to split the activations into blocks of approximately
  # equal size, with none larger than max_block_size.
  n_r = tf.shape(input=real_activations)[0]
  n_g = tf.shape(input=generated_activations)[0]

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

  dim = tf.cast(real_activations.shape[1], dtype)

  def compute_kid_block(i):
    """Computes the ith block of the KID estimate."""
    r_s = inds_r[i]
    r_e = inds_r[i + 1]
    r = real_activations[r_s:r_e]
    m = tf.cast(r_e - r_s, dtype)

    g_s = inds_g[i]
    g_e = inds_g[i + 1]
    g = generated_activations[g_s:g_e]
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
