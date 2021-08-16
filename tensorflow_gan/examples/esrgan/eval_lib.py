# coding=utf-8
# Copyright 2021 The TensorFlow GAN Authors.
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

import tensorflow as tf
from absl import logging
import utils
import collections

HParams = collections.namedtuple('HParams', [
    'batch_size', 'hr_dimension',
    'scale',
    'model_dir', 'data_dir',
    'num_steps', 'num_inception_images', 
    'image_dir', 'eval_real_images'])

def evaluate(hparams, generator, data):
  """ Runs an evaluation loop and calculates the mean FID,
      Inception and PSNR scores observed on the validation dataset.

  Args:
      hparams: Parameters for evaluation.
      generator : The trained generator network.
      data : Validation DIV2K dataset.
  """
  fid_metric = tf.keras.metrics.Mean()
  inc_metric = tf.keras.metrics.Mean()
  psnr_metric = tf.keras.metrics.Mean()
  step = 0
  for lr, hr in data.take(hparams.num_steps):
    step += 1
    # Generate fake images for evaluating the model
    gen = generator(lr)

    if step % hparams.num_steps//10:
      utils.visualize_results(lr,
                              gen,
                              hr,
                              image_dir=hparams.image_dir,
                              step=step)

    # Compute Frechet Inception Distance.
    fid_score = utils.get_frechet_inception_distance(
        hr, gen,
        hparams.batch_size,
        hparams.num_inception_images)

    fid_metric(fid_score)

    # Compute Inception Scores.
    if hparams.eval_real_images:
      inc_score = utils.get_inception_scores(hr,
                                             hparams.batch_size,
                                             hparams.num_inception_images)
    else:
      inc_score = utils.get_inception_scores(gen,
                                             hparams.batch_size,
                                             hparams.num_inception_images)
    inc_metric(inc_score)

    # Compute PSNR values.
    psnr = utils.get_psnr(hr, gen)
    psnr_metric(psnr)

  logging.info('FID Score :{}\tInception Score :{}\tPSNR value :{}'.format(
      fid_metric.result(), inc_metric.result(), psnr_metric.result()))