<!-- TODO(joelshor): Add images to the examples. -->
# TensorFlow-GAN (TF-GAN)

TF-GAN is a lightweight library for training and evaluating Generative
Adversarial Networks (GANs). GANs allow you to train a network (called a
'generator') to generate samples from a distribution, without having to
explicitly model the distribution and without writing an explicit loss. For
example, the generator could learn to draw samples from the distribution of
natural images. The generator learns by incorporating feedback from a model
(called a 'discriminator') that tries to distinguish between samples created by
the generator and samples in the training data.

For more details on this technique, see ['Generative Adversarial
Networks'](https://arxiv.org/abs/1406.2661) by Goodfellow et al. See
[the examples directory](https://github.com/tensorflow/gan/examples/) for examples, and [this
tutorial](https://github.com/tensorflow/gan) for an introduction.

## Usage
```python
import tensorflow_gan as tfgan
```

## Why TF-GAN?

* Easily train generator and discriminator networks with well-tested, flexible [library calls](https://github.com/tensorflow/gan/python/train.py). You can
mix TF-GAN, native TF, and other custom frameworks.
* Use TF-GAN's implementations of [GAN losses and penalties](https://github.com/tensorflow/gan/python/losses/python/losses_impl.py),
  such as  Wasserstein loss, gradient penalty, mutual information penalty, etc.
* Use TensorFlow tools to [monitor and
  visualize](https://github.com/tensorflow/gan/python/eval/python/summaries_impl.py) GAN progress
  during training, and to [evaluate](https://github.com/tensorflow/gan/python/eval/python/classifier_metrics_impl.py)
  the trained model.
* Use TF-GAN's implementations of
  [techniques](https://github.com/tensorflow/gan/python/features/python/)
  to stabilize and improve training.
* Develop based on examples of [common GAN setups](https://github.com/tensorflow/gan/examples).
* Use the TF-GAN-backed [GANEstimator](https://github.com/tensorflow/gan/python/estimator/python/gan_estimator_impl.py) to easily train a GAN model.
* Benefit from improvements in TF-GAN infrastructure.
* Stay up-to-date with research as we add more algorithms.

## Structure of the TF-GAN Library

TF-GAN is composed of several parts, which were designed to exist independently:

*   [core](https://github.com/tensorflow/gan/python/train.py):
    the main infrastructure needed to train a GAN. Set up training with
    any combination of custom-code and TF-GAN library calls. More details
    [below](#training).

*   [features](https://github.com/tensorflow/gan/python/features/):
    well-tested implementations of many common GAN operations and
    normalization techniques, such as instance normalization and conditioning.

*   [losses](https://github.com/tensorflow/gan/python/losses/):
    well-tested implementations of losses and
    penalties, such as the Wasserstein loss, gradient penalty, mutual
    information penalty, etc.

*   [evaluation](https://github.com/tensorflow/gan/python/eval/):
    well-tested implementations of standard GAN evaluation metrics.
    Use `Inception Score`, `Frechet Distance`, or `Kernel Distance` with a
    pretrained Inception network to evaluate your unconditional generative
    model. You can also use your own pretrained classifier for more specific
    performance numbers, or use other methods for evaluating conditional
    generative models.

*   [examples](https://github.com/tensorflow/gan/)
    and [tutorial](https://github.com/tensorflow/gan): examples of how to use TF-GAN
    to make GAN training easier, as well as examples of more complicated GAN
    setups. These include unconditional and conditional
    GANs, InfoGANs, adversarial losses on existing networks, and image-to-image
    translation.

## Training a GAN model <a id="training"></a>

Training in TF-GAN typically consists of the following steps:

1. Specify the input to your networks.
1. Set up your generator and discriminator using a `GANModel`.
1. Specify your loss using a `GANLoss`.
1. Create your train ops using a `GANTrainOps`.
1. Run your train ops.

At each stage, you can either use TF-GAN's convenience functions, or you can
perform the step manually for fine-grained control.

There are various types of GAN setup. For instance, you can train a generator
to sample unconditionally from a learned distribution, or you can condition on
extra information such as a class label. TF-GAN is compatible with many setups,
and we demonstrate a few below:

### Examples

#### Unconditional MNIST generation

This example trains a generator to produce handwritten MNIST digits. The
generator maps random draws from a multivariate normal distribution to MNIST
digit images. See ['Generative Adversarial
Networks'](https://arxiv.org/abs/1406.2661) by Goodfellow et al.

```python
# Set up the input.
images = mnist_data_provider.provide_data(FLAGS.batch_size)
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# Set up the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=mnist.unconditional_generator,  # you define
    discriminator_fn=mnist.unconditional_discriminator,  # you define
    real_data=images,
    generator_inputs=noise)

# Specify the GAN loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)

# Create the train ops, which calculate gradients and apply updates to weights.
train_ops = tfgan.gan_train_ops(
    gan_model,
    gan_loss,
    generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5))

# Run the train ops in the alternating training scheme.
tfgan.gan_train(
    train_ops,
    hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps)],
    logdir=FLAGS.train_log_dir)
```

#### Conditional MNIST generation
This example trains a generator to generate MNIST images *of a given class*.
The generator maps random draws from a multivariate normal distribution and a
one-hot label of the desired digit class to an MNIST digit image. See
['Conditional Generative Adversarial Nets'](https://arxiv.org/abs/1411.1784) by
Mirza and Osindero.

```python
# Set up the input.
images, one_hot_labels = mnist_data_provider.provide_data(FLAGS.batch_size)
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# Set up the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=mnist.conditional_generator,  # you define
    discriminator_fn=mnist.conditional_discriminator,  # you define
    real_data=images,
    generator_inputs=(noise, one_hot_labels))

# The rest is the same as in the unconditional case.
...
```
#### Adversarial loss
This example combines an L1 pixel loss and an adversarial loss to learn to
autoencode images. The bottleneck layer can be used to transmit compressed
representations of the image. Neutral networks with pixel-wise loss only tend to
produce blurry results, so the GAN can be used to make the reconstructions more
plausible. See ['Full Resolution Image Compression with Recurrent Neural Networks'](https://arxiv.org/abs/1608.05148) by Toderici et al
for an example of neural networks used for image compression, and ['Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network'](https://arxiv.org/abs/1609.04802) by Ledig et al for a more detailed description of
how GANs can sharpen image output.

```python
# Set up the input pipeline.
images = image_provider.provide_data(FLAGS.batch_size)

# Set up the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=nets.autoencoder,  # you define
    discriminator_fn=nets.discriminator,  # you define
    real_data=images,
    generator_inputs=images)

# Specify the GAN loss and standard pixel loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    gradient_penalty=1.0)
l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

# Modify the loss tuple to include the pixel loss.
gan_loss = tfgan.losses.combine_adversarial_loss(
    gan_loss, gan_model, l1_pixel_loss, weight_factor=FLAGS.weight_factor)

# The rest is the same as in the unconditional case.
...
```

#### Image-to-image translation This example maps images in one domain to images
of the same size in a different dimension. For example, it can map segmentation
masks to street images, or grayscale images to color. See ['Image-to-Image
Translation with Conditional Adversarial
Networks'](https://arxiv.org/abs/1611.07004) by Isola et al for more details.

```python
# Set up the input pipeline.
input_image, target_image = data_provider.provide_data(FLAGS.batch_size)

# Set up the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=nets.generator,  # you define
    discriminator_fn=nets.discriminator,  # you define
    real_data=target_image,
    generator_inputs=input_image)

# Set up the GAN loss and standard pixel loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.least_squares_generator_loss,
    discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss)
l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

# Modify the loss tuple to include the pixel loss.
gan_loss = tfgan.losses.combine_adversarial_loss(
    gan_loss, gan_model, l1_pixel_loss, weight_factor=FLAGS.weight_factor)

# The rest is the same as in the unconditional case.
...
```

#### InfoGAN

Train a generator to generate specific MNIST digit images, and control for digit
style *without using any labels*. See ['InfoGAN: Interpretable Representation
Learning by Information Maximizing Generative Adversarial
Nets'](https://arxiv.org/abs/1606.03657) for more details.

```python
# Set up the input pipeline.
images = mnist_data_provider.provide_data(FLAGS.batch_size)

# Set up the generator and discriminator.
gan_model = tfgan.infogan_model(
    generator_fn=mnist.infogan_generator,  # you define
    discriminator_fn=mnist.infogran_discriminator,  # you define
    real_data=images,
    unstructured_generator_inputs=unstructured_inputs,  # you define
    structured_generator_inputs=structured_inputs)  # you define

# Set up the GAN loss with mutual information penalty.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    gradient_penalty=1.0,
    mutual_information_penalty_weight=1.0)

# The rest is the same as in the unconditional case.
...
```

#### Custom model creation
Train an unconditional GAN to generate MNIST digits, but manually construct
the `GANModel` tuple for more fine-grained control.

```python
# Set up the input pipeline.
images = mnist_data_provider.provide_data(FLAGS.batch_size)
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# Manually build the generator and discriminator.
with tf.variable_scope('Generator') as gen_scope:
  generated_images = generator_fn(noise)
with tf.variable_scope('Discriminator') as dis_scope:
  discriminator_gen_outputs = discriminator_fn(generated_images)
with tf.variable_scope(dis_scope, reuse=True):
  discriminator_real_outputs = discriminator_fn(images)
generator_variables = get_trainable_variables(gen_scope)
discriminator_variables = get_trainable_variables(dis_scope)
# Depending on what TF-GAN features you use, you don't always need to supply
# every `GANModel` field. At a minimum, you need to include the discriminator
# outputs and variables if you want to use TF-GAN to construct losses.
gan_model = tfgan.GANModel(
    generator_inputs,
    generated_data,
    generator_variables,
    gen_scope,
    generator_fn,
    real_data,
    discriminator_real_outputs,
    discriminator_gen_outputs,
    discriminator_variables,
    dis_scope,
    discriminator_fn)

# The rest is the same as the unconditional case.
...
```

## Authors
Joel Shor (github: [joel-shor](https://github.com/joel-shor))
