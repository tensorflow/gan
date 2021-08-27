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

"""Train the ESRGAN model
See https://arxiv.org/abs/1809.00219 for details about the model.
"""

import tensorflow as tf
import collections
import os
from absl import logging

import networks
import losses
import utils

hparams = collections.namedtuple('hparams', [
    'batch_size', 'scale',
    'model_dir', 'phase_1',
    'phase_2', 'hr_dimension',
    'data_dir', 'print_steps',
    'trunk_size', 'total_steps',
    'decay_steps', 'decay_factor',
    'lr', 'beta_1',
    'beta_2', 'init_lr',
    'loss_type', 'lambda_',
    'eta', 'image_dir'])

def pretrain_generator(hparams, data):
  """ Pre-trains the generator network with pixel-loss as proposed in
      the paper and saves the network inside the model directory specified.

  Args:
      hparams : Training parameters as proposed in the paper.
      data : Dataset consisting of LR and HR image pairs.
  """

  # Stores mean L1 values and PSNR values obtained during training.
  metric = tf.keras.metrics.Mean()
  psnr_metric = tf.keras.metrics.Mean()

  # If phase_1 training is done, load that generator model.
  if hparams.phase_1:
    generator = tf.keras.load_model(hparams.model_dir + '/Phase_1/generator/')
  # If pre-trained model is not available, start training from the beginning
  else:
    generator = networks.generator_network(hparams)

  logging.info("Starting Phase-1 training of generator using only pixel loss function.")

  g_optimizer = _get_optimizer()

  def train_step(image_lr, image_hr):
    """ Calculates the L1 Loss and gradients at each step, and updates the
        gradient to improve the PSNR values.
    Args:
        image_lr : batch of tensors representing LR images.
        image_hr : batch of tensors representing HR images.

    Returns:
        PSNR values and generator loss obtained in each step.
    """
    with tf.GradientTape() as tape:
      fake = generator(image_lr)

      gen_loss = losses.pixel_loss(image_hr, fake) * (1.0 / hparams.batch_size)
      psnr = utils.get_psnr(image_hr, fake)

      gradient = tape.gradient(gen_loss, generator.trainable_variables)
      g_optimizer.apply_gradients(zip(gradient, generator.trainable_variables))

      return psnr, gen_loss

  step = 0
  for lr, hr in data.take(hparams.total_steps):
    step += 1
    lr = tf.cast(lr, tf.float32)
    hr = tf.cast(hr, tf.float32)

    psnr, gen_loss = train_step(lr, hr)

    # Calculate the mean loss and PSNR values obtained during training.
    metric(gen_loss)
    psnr_metric(psnr)

    if step % hparams.print_steps == 0:
      logging.info("Step: {}\tGenerator Loss: {}\tPSNR: {}".format(
          step, metric.result(), psnr_metric.result()))

    # Modify the learning rate as mentioned in the paper.
    if step % hparams.decay_steps == 0:
      g_optimizer.learning_rate.assign(
          g_optimizer.learning_rate * hparams.decay_factor)

  # Save the generator model inside model_dir.
  os.makedirs(hparams.model_dir + '/Phase_1/generator', exist_ok=True)
  generator.save(hparams.model_dir + '/Phase_1/generator')
  logging.info("Saved pre-trained generator network succesfully!")

def train_esrgan(hparams, data):
  """Loads the pre-trained generator model and trains the ESRGAN network
     using L1 Loss, Perceptual loss and RaGAN loss function.

  Args:
      hparams : Training parameters as proposed in the paper.
      data : Dataset consisting of LR and HR image pairs.
  """
  # If the phase 2 training is done,load thd trained networks.
  if hparams.phase_2:
    generator = tf.keras.models.load_model(hparams.model_dir +
                                           'Phase_2/generator/')
    discriminator = tf.keras.models.load_model(hparams.model_dir +
                                               'Phase_2/discriminator/')
  # If Phase 2 training is not done, then load the pre-trained generator model.
  else:
    try:
      generator = tf.keras.models.load_model(hparams.model_dir +
                                             '/Phase_1/generator')
    except:
      raise FileNotFoundError('Pre-trained Generator model not found!')


    discriminator = networks.discriminator_network(hparams)

  logging.info("Starting Phase-2 training of ESRGAN")

  # Generator learning rate is set as 1 x 10^-4.
  g_optimizer = _get_optimizer(lr=hparams.init_lr)
  d_optimizer = _get_optimizer()

  # Define the Perceptual loss function and
  # pass 'imagenet' as the weight for the VGG-19 network.
  perceptual_loss = losses.vgg_loss(
      weight="imagenet",
      input_shape=[hparams.hr_dimension, hparams.hr_dimension, 3])

  gen_metric = tf.keras.metrics.Mean()
  disc_metric = tf.keras.metrics.Mean()
  psnr_metric = tf.keras.metrics.Mean()

  def train_step(image_lr, image_hr, step):
    """ Calculates the L1 Loss, Perceptual loss and RaGAN loss, to train
        both generator and discriminator networks of the ESRGAN model.
    Args :
        image_lr : batch of tensors representing LR images.
        image_hr : batch of tensors representing HR images.

    Returns :
        PSNR values, generator loss and discriminator obtained in each step.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen = generator(image_lr)

      fake = utils.preprocess_input(gen)
      image_lr = utils.preprocess_input(image_lr)
      image_hr = utils.preprocess_input(image_hr)

      percep_loss = tf.reduce_mean(perceptual_loss(image_hr, fake))
      l1_loss = losses.pixel_loss(image_hr, fake)
      
      # TODO(Nived): Replace these functions once TF-GAN losses are updated. 
      loss_RaG = losses.ragan_generator_loss(discriminator(image_hr), 
                                             discriminator(fake))
      disc_loss = losses.ragan_discriminator_loss(discriminator(image_hr), 
                                                  discriminator(fake))

      gen_loss = percep_loss + hparams.lambda_ * loss_RaG + hparams.eta * l1_loss

      gen_loss = gen_loss * (1.0 / hparams.batch_size)
      disc_loss = disc_loss * (1.0 / hparams.batch_size)      
      psnr = utils.get_psnr(image_hr, fake)


      disc_grad = disc_tape.gradient(disc_loss,
                                     discriminator.trainable_variables)
      d_optimizer.apply_gradients(zip(disc_grad,
                                      discriminator.trainable_variables))

      gen_grad = gen_tape.gradient(gen_loss,
                                   generator.trainable_variables)
      g_optimizer.apply_gradients(zip(gen_grad,
                                      generator.trainable_variables))

      return gen_loss, disc_loss, psnr

  def val_step(image_lr, image_hr, step):
    """ Saves an image grid containing LR image, generated image and
        HR image, inside the image directory.
    Args:
        image_lr : Low Resolution Image 
        image_hr : High Resolution Image. 
        step : Number of steps completed, used for naming the image 
               file. 
    """
    fake=generator(image_lr)
    utils.visualize_results(image_lr, 
                            fake, 
                            image_hr,
                            hparams.image_dir,
                            step=step)

  step = 0
  # Modify learning rate at each of these steps
  decay_list = [50000, 100000, 200000, 300000]
  index=0

  for lr, hr in data.take(hparams.total_steps):
    step += 1
    lr = tf.cast(lr, tf.float32)
    hr = tf.cast(hr, tf.float32)

    gen_loss, disc_loss, psnr = train_step(lr, hr, step)

    gen_metric(gen_loss)
    disc_metric(disc_loss)
    psnr_metric(psnr)

    if step % hparams.print_steps == 0:
      logging.info("Step: {}\tGenerator Loss: {}\tDiscriminator: {}\tPSNR: {}"
                   .format(step, gen_metric.result(), disc_metric.result(),
                           psnr_metric.result()))
      val_step(lr, hr, step)

    # Modify the learning rate as mentioned in the paper.
    if step >= decay_list[index]:
      g_optimizer.learning_rate.assign(
          g_optimizer.learning_rate * hparams.decay_factor)
      d_optimizer.learning_rate.assign(
          d_optimizer.learning_rate * hparams.decay_factor)
  
      index+=1
      

  # Save the generator model inside model_dir.
  os.makedirs(hparams.model_dir + '/Phase_2/generator', exist_ok=True)
  generator.save(hparams.model_dir + '/Phase_2/generator')
  logging.info("Saved trained ESRGAN generator succesfully!")

  interpolated_generator = utils.network_interpolation(
      phase_1_path=hparams.model_dir + '/Phase_1/generator',
      phase_2_path=hparams.model_dir + '/Phase_2/generator')

  #Save interpolated generator
  os.makedirs(hparams.model_dir 
              + '/Phase_2/interpolated_generator', exist_ok=True)
  interpolated_generator.save(hparams.model_dir 
                              + '/Phase_2/interpolated_generator')

  logging.info("Saved interpolated generator network succesfully!")

def _get_optimizer(lr=0.0002, beta_1=0.9, beta_2=0.99):
  """Returns the Adam optimizer with the specified learning rate."""
  optimizer =  tf.optimizers.Adam(
                        learning_rate=lr,
                        beta_1=beta_1,
                        beta_2=beta_2)
  return optimizer
