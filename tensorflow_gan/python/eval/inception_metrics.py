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

r"""Model evaluation tools for TF-GAN.

These methods come from https://arxiv.org/abs/1606.03498 and
https://arxiv.org/abs/1706.08500.

NOTE: This implementation uses the same weights as in
https://github.com/openai/improved-gan/blob/master/inception_score/model.py,
.


Note that the default checkpoint is the same as in the OpenAI implementation
(https://github.com/openai/improved-gan/tree/master/inception_score), but is
more numerically stable and is an unbiased estimator of the true Inception score
even when splitting the inputs into batches. Also, the graph modified so that it
works with arbitrary batch size and the preprocessing moved to the `preprocess`
function. Note that the modifications in the GitHub implementation are *not*
sufficient to run with arbitrary batch size, due to the hardcoded resize value.

The graph runs on TPU.

Finally, I manually removed the placeholder input, which was unnecessary and is
not supported on TPU.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

import tensorflow as tf
from tensorflow_gan.python.eval import classifier_metrics
import tensorflow_hub as tfhub


__all__ = [
    'classifier_fn_from_tfhub',
    'run_inception',
    'sample_and_run_inception',
    'inception_score',
    'inception_score_streaming',
    'frechet_inception_distance',
    'frechet_inception_distance_streaming',
    'kernel_inception_distance',
    'kernel_inception_distance_and_std',
    'INCEPTION_TFHUB',
    'INCEPTION_OUTPUT',
    'INCEPTION_FINAL_POOL',
    'INCEPTION_DEFAULT_IMAGE_SIZE',
]

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {INCEPTION_OUTPUT: tf.float32,
                   INCEPTION_FINAL_POOL: tf.float32}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def classifier_fn_from_tfhub(tfhub_module, output_fields, return_tensor=False):
  """Returns a function that can be as a classifier function.

  Wrapping the TF-Hub module in another function defers loading the module until
  use, which is useful for mocking and not computing heavy default arguments.

  Args:
    tfhub_module: A string handle for a TF-Hub module.
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]
  def _classifier_fn(images):
    output = tfhub.load(tfhub_module)(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)
  return _classifier_fn


run_inception = functools.partial(
    classifier_metrics.run_classifier_fn,
    classifier_fn=classifier_fn_from_tfhub(INCEPTION_TFHUB, None),
    dtypes=_DEFAULT_DTYPES)


sample_and_run_inception = functools.partial(
    classifier_metrics.sample_and_run_classifier_fn,
    classifier_fn=classifier_fn_from_tfhub(INCEPTION_TFHUB, None),
    dtypes=_DEFAULT_DTYPES)

inception_score = functools.partial(
    classifier_metrics.classifier_score,
    classifier_fn=classifier_fn_from_tfhub(
        INCEPTION_TFHUB, INCEPTION_OUTPUT, True))

inception_score_streaming = functools.partial(
    classifier_metrics.classifier_score_streaming,
    classifier_fn=classifier_fn_from_tfhub(
        INCEPTION_TFHUB, INCEPTION_OUTPUT, True))

frechet_inception_distance = functools.partial(
    classifier_metrics.frechet_classifier_distance,
    classifier_fn=classifier_fn_from_tfhub(
        INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True))

frechet_inception_distance_streaming = functools.partial(
    classifier_metrics.frechet_classifier_distance_streaming,
    classifier_fn=classifier_fn_from_tfhub(
        INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True))

kernel_inception_distance = functools.partial(
    classifier_metrics.kernel_classifier_distance,
    classifier_fn=classifier_fn_from_tfhub(
        INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True))

kernel_inception_distance_and_std = functools.partial(
    classifier_metrics.kernel_classifier_distance_and_std,
    classifier_fn=classifier_fn_from_tfhub(
        INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True))
