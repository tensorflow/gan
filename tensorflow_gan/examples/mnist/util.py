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

"""A utility for evaluating MNIST generative models.

These functions use a pretrained MNIST classifier with ~99% eval accuracy to
measure various aspects of the quality of generated MNIST digits.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow_gan as tfgan

__all__ = [
    'mnist_score',
    'mnist_frechet_distance',
    'mnist_cross_entropy',
]

# The references to `MODEL_GRAPH_DEF` below are removed in open source by a
# copy bara transformation..
# Prepend `../`, since paths start from `third_party/tensorflow`.
MODEL_GRAPH_DEF = '../py/tensorflow_gan/examples/mnist/data/classify_mnist_graph_def.pb'
# The open source code finds the graph def by relative filepath.
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_BY_FN = os.path.join(CUR_DIR, 'data', 'classify_mnist_graph_def.pb')

INPUT_TENSOR = 'inputs:0'
OUTPUT_TENSOR = 'logits:0'


def mnist_score(images,
                graph_def_filename=None,
                input_tensor=INPUT_TENSOR,
                output_tensor=OUTPUT_TENSOR,
                num_batches=1):
  """Get MNIST classifier score.

  Args:
    images: A minibatch tensor of MNIST digits. Shape must be [batch, 28, 28,
      1].
    graph_def_filename: Location of a frozen GraphDef binary file on disk. If
      `None`, uses a default graph.
    input_tensor: GraphDef's input tensor name.
    output_tensor: GraphDef's output tensor name.
    num_batches: Number of batches to split `generated_images` in to in order to
      efficiently run them through Inception.

  Returns:
    The classifier score, a floating-point scalar.
  """
  images.shape.assert_is_compatible_with([None, 28, 28, 1])

  graph_def = _graph_def_from_par_or_disk(graph_def_filename)
  mnist_classifier_fn = lambda x: tfgan.eval.run_image_classifier(  # pylint: disable=g-long-lambda
      x, graph_def, input_tensor, output_tensor)

  score = tfgan.eval.classifier_score(images, mnist_classifier_fn, num_batches)
  score.shape.assert_is_compatible_with([])

  return score


def mnist_frechet_distance(real_images,
                           generated_images,
                           graph_def_filename=None,
                           input_tensor=INPUT_TENSOR,
                           output_tensor=OUTPUT_TENSOR,
                           num_batches=1):
  """Frechet distance between real and generated images.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Please see TF-GAN for implementation details.

  Args:
    real_images: Real images to use to compute Frechet Inception distance.
    generated_images: Generated images to use to compute Frechet Inception
      distance.
    graph_def_filename: Location of a frozen GraphDef binary file on disk. If
      `None`, uses a default graph.
    input_tensor: GraphDef's input tensor name.
    output_tensor: GraphDef's output tensor name.
    num_batches: Number of batches to split images into in order to efficiently
      run them through the classifier network.

  Returns:
    The Frechet distance. A floating-point scalar.
  """
  real_images.shape.assert_is_compatible_with([None, 28, 28, 1])
  generated_images.shape.assert_is_compatible_with([None, 28, 28, 1])

  graph_def = _graph_def_from_par_or_disk(graph_def_filename)
  mnist_classifier_fn = lambda x: tfgan.eval.run_image_classifier(  # pylint: disable=g-long-lambda
      x, graph_def, input_tensor, output_tensor)

  frechet_distance = tfgan.eval.frechet_classifier_distance(
      real_images, generated_images, mnist_classifier_fn, num_batches)
  frechet_distance.shape.assert_is_compatible_with([])

  return frechet_distance


def mnist_cross_entropy(images,
                        one_hot_labels,
                        graph_def_filename=None,
                        input_tensor=INPUT_TENSOR,
                        output_tensor=OUTPUT_TENSOR):
  """Returns the cross entropy loss of the classifier on images.

  Args:
    images: A minibatch tensor of MNIST digits. Shape must be [batch, 28, 28,
      1].
    one_hot_labels: The one hot label of the examples. Tensor size is [batch,
      10].
    graph_def_filename: Location of a frozen GraphDef binary file on disk. If
      `None`, uses a default graph embedded in the par file.
    input_tensor: GraphDef's input tensor name.
    output_tensor: GraphDef's output tensor name.

  Returns:
    A scalar Tensor representing the cross entropy of the image minibatch.
  """
  graph_def = _graph_def_from_par_or_disk(graph_def_filename)

  logits = tfgan.eval.run_image_classifier(images, graph_def, input_tensor,
                                           output_tensor)
  return tf.compat.v1.losses.softmax_cross_entropy(
      one_hot_labels, logits, loss_collection=None)


def _graph_def_from_par_or_disk(filename):
  if filename is None:
    return tfgan.eval.get_graph_def_from_disk(MODEL_BY_FN)
  else:
    return tfgan.eval.get_graph_def_from_disk(filename)
