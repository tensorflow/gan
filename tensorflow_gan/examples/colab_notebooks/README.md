## CIFAR10 on Cloud TPU

Author: Tom Brown, Joel Shor, David Westbrook

### How to run


You can run the introductory tutorial colab notebook
[here](https://github.com/tensorflow/gan/examples/colab_notebooks/tfgan_tutorial.ipynb).
You can run the cloud TPU notebook
[here](https://github.com/tensorflow/gan/examples/colab_notebooks/tfgan_on_tpus.ipynb).

### Description

A colaboratory notebook will introduce you to using TF-GAN's `GANEstimator`.
The estimator framework abstracts the training details so
you can focus on the details that matter.

A second colaboratory notebook will introduce you to using
TF-GAN's `TPUGANEstimator` to train GANs on Google's cloud TPU. This
infrastructure gives you unprecedented compute power and batch size. In
less than **five minutes**, you can train an unconditional GAN on CIFAR10.

<img src="images/cifar_cloudtpu.png" title="GAN on Cloud TPU" width="330"/>
