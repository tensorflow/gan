## CIFAR10

Author: Joel Shor

### How to run


1. Run the setup instructions in `tensorflow_gan/examples/README.md`
1. Run:

```
python cifar/train.py
```

### Description

We train a [DCGAN generator](https://arxiv.org/abs/1511.06434) to produce [CIFAR10 images](https://www.cs.toronto.edu/~kriz/cifar.html).
The unconditional case maps noise to CIFAR10 images. The conditional case maps
noise and image class to CIFAR10 images. We use the [Inception Score](https://arxiv.org/abs/1606.03498)
and a CIFAR10 classifier cross-entropy to evaluate the images.

#### Unconditional CIFAR10
<img src="images/cifar_unconditional_gan.png" title="Unconditional GAN" width="330" />

#### Conditional CIFAR10
<img src="images/cifar_conditional_gan.png" title="Conditional GAN" width="330" />
