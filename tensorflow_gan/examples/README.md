# TF-GAN Examples

[TF-GAN](https://github.com/tensorflow/gan/tree/master/tensorflow_gan)
is a lightweight library for training and evaluating Generative Adversarial
Networks (GANs). GANs have been in a wide range of tasks including
[image translation](https://arxiv.org/abs/1703.10593),
[superresolution](https://arxiv.org/abs/1609.04802), and
[data augmentation](https://arxiv.org/abs/1612.07828). This directory contains
fully-working examples. Each subdirectory contains a different working example.
The sub-sections below describe each of the problems, and include some sample
outputs. **Be sure to follow the instructions for how to run the examples.**

## Steps to run an example

1.  Add the examples directory to your PYTHONPATH environment variable with

    ```shell
    export PYTHONPATH=${TFGAN_REPO}/tensorflow_gan/examples:${PYTHONPATH}
    ```

    Be sure to use the location where you cloned this repository.
1.  Add this repository to your PYTHONPATH environment variable so that it can
    be used for `tensorflow_gan` instead of any older libraries you might have
    installed.

    ```shell
    export PYTHONPATH=${TFGAN_REPO}:${PYTHONPATH}
    ```

1.  Install the necessary dependencies, which depend on which example you want
    to run. At a minimum, you will need:
    1. `tensorflow`
    1. `tensorflow_probability`
    1. `tensorflow_datasets`

    You can usually install these with:

    ```shell
    pip install --upgrade $PACKAGE --user
    ```

1.  Follow the instructions in the particular example directory's `README.md`.

## Debugging

1.  If you get an error like `ImportError: No module named xxx`, you might not
    have set the `PYTHONPATH` properly. See step #1 above.
1.  If you get an error for something basic like
    `ImportError: No module named absl`, you might not have installed all the
    imports you need.

## Steps to add an example

1.  Email joelshor@google.com to propose the idea.
1.  Add a `README.md` to your new subdirectory. Be sure to include a "How to
    run" section.
1.  Add a subdirectory with output ex "images" or "audio".
1.  Add a line and high-level summary to this file.
1.  Submit it and profit.

## Table of contents

1.  [MNIST](#mnist)
1.  [MNIST Estimator](#mnist_estimator)
1.  [CIFAR10](#cifar10)
1.  [CIFAR10 on Cloud TPU](#cifar10_tpu)
1.  [CycleGAN](#cyclegan)
1.  [StarGAN](#stargan)
1.  [StarGAN Estimator](#stargan_estimator)
1.  [Progressive GAN](#progressive_gan)
1.  [Self-Attention GAN](#self_attention_gan)

## MNIST
<a id='mnist'></a>

Authors: Joel Shor, Yoel Drori

An unconditional GAN, a conditional GAN, and an
[InfoGAN](https://arxiv.org/abs/1606.03657), all trained on
[MNIST digits](http://yann.lecun.com/exdb/mnist/). We use a classifier trained
on MNIST digit classification for evaluation.

## MNIST Estimator
<a id='mnist_estimator'></a>

Author: Joel Shor

Two examples. Both are unconditional GANs trained on MNIST using the
`tfgan.estimator.GANEstimator`,
which reduces code complexity and abstracts away the training details.
The first uses the `tf.Estimator` "blessed" method using `train_and_evaluate`.
The second example uses custom estimator calls.

## CIFAR10
<a id='cifar10'></a>

Author: Joel Shor

We train a [DCGAN generator](https://arxiv.org/abs/1511.06434) to produce [CIFAR10 images](https://www.cs.toronto.edu/~kriz/cifar.html).
The unconditional case maps noise to CIFAR10 images. The conditional case maps
noise and image class to CIFAR10 images. We use the [Inception Score](https://arxiv.org/abs/1606.03498) to evaluate the
images.

## CIFAR10 on Cloud TPU
<a id='cifar10_tpu'></a>

Authors: Tom Brown, Joel Shor, David Westbrook

This [colaboratory notebook](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/examples/colab_notebooks/tfgan_on_tpus.ipynb)
introduces you to using
TF-GAN's `TPUGANEstimator` to train GANs on Google's cloud TPU. This
infrastructure gives you unprecedented compute power and batch size. In less
than **five minutes**, you can train an unconditional GAN on CIFAR10.

## CycleGAN
<a id='cyclegan'></a>

Author: Shuo Chen and Cassandra Xia

Based on the paper ["Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks"](https://arxiv.org/abs/1703.10593), this example converts
one set of images into another set in an unpaired way.

## StarGAN

<a id='stargan'></a>

Author: Wesley Qian

A [StarGAN](https://arxiv.org/abs/1711.09020) implementation for multi-domain
image translation. We run StarGAN on the
[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## StarGAN Estimator

<a id='stargan_estimator'></a>

Author: Wesley Qian

An Estimator implementation of the example above using
`tfgan.estimator.StarGANEstimator`.

## Progressive GAN
<a id='progressive_gan'></a>

Author: Shuo Chen

An implementation of the technique described in
[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).
We run the network on the CIFAR10 dataset.

## Self-Attention GAN
<a id='self_attention_gan'></a>

Author: Yoel Drori, Augustus Odena, Joel Shor

An implementation of
[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318).
The generator produces downsampled ImageNet images. It can be trained on GPU
or cloud TPU.
