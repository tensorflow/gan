## Progressive GAN

Author: Shuo Chen

### How to run


1. Run the setup instructions in [tensorflow_gan/examples/README.md](https://github.com/tensorflow/gan/blob/master/tensorflow_gan/examples/README.md#steps-to-run-an-example)
1. Install matplotlib by executing `pip install matplotlib`
1. Run:

```python
python progressive_gan/train_main.py --alsologtostderr
```

### Description

An implementation of the technique described in
[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).
We run the network on the CIFAR10 dataset.

<img src="images/progressive_gan_20m.png" title="Unconditional GAN" width="330" />
