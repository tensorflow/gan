## MNIST

Author: Joel Shor

### How to run


1.  Run the setup instructions in [tensorflow_gan/examples/README.md](https://github.com/tensorflow/gan/blob/master/tensorflow_gan/examples/README.md#steps-to-run-an-example)
1.  Run:

    ```shell
    python mnist/train.py --gan_type=[TYPE] --train_log_dir=/tmp/mnist
    ```

    here `[TYPE]` can be is one of `unconditional`, `conditional` or `infogan`.

### Description

We train a simple generator to produce [MNIST digits](http://yann.lecun.com/exdb/mnist/).
The unconditional case maps noise to MNIST digits. The conditional case maps
noise and digit class to MNIST digits.
[InfoGAN](https://arxiv.org/abs/1606.03657) learns to produce
digits of a given class without labels, as well as controlling style.
We use a classifier trained on MNIST digit classification for evaluation.

Unconditional | Conditional | InfoGAN
-------              | --------------                        | --------           |  --------
<img src="images/mnist_unconditional_gan.png" title="Unconditional GAN" width="330" /> | <img src="images/mnist_conditional_gan.png" title="Conditional GAN" width="330" /> | <img src="images/mnist_infogan.png" title="InfoGAN" width="330" />
