# MNIST Estimator

Author: Joel Shor

### How to run


1. Run the setup instructions in `tensorflow_gan/examples/README.md`
1. Install: `scipy`.
1. Run:

```
python mnist_estimator/train.py --max_number_of_steps_mnist_estimator=20000 --output_dir_mnist_estimator=/tmp/mnist-estimator --alsologtostderr
```

### Description

The Estimator setup is exactly the same, but uses the
`tfgan.estimator.GANEstimator` to reduce code complexity and abstract away the
training details.

<img src="images/mnist_estimator_unconditional_gan.png" title="Unconditional GAN" width="330" />
