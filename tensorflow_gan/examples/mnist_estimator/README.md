# MNIST Estimator

Author: Joel Shor

### How to run


'Blessed' version using 'train_and_evaluate':

1. Run the setup instructions in `tensorflow_gan/examples/README.md`
1. Install: `scipy`.
1. Run:

```
python mnist_estimator/train_experiment.py --max_number_of_steps=20000 --output_dir=/tmp/mnist-estimator-tae --alsologtostderr
```

Using custom estimator calls:

1. Run the setup instructions in `tensorflow_gan/examples/README.md`
1. Install: `scipy`.
1. Run:

```
python mnist_estimator/train.py --max_number_of_steps=20000 --output_dir=/tmp/mnist-estimator --alsologtostderr
```

### Description

This folder contains two related examples. One uses the simple, `tf.Estimator`
"blessed" method of running `tf.estimator.train_and_evaluate`. This abstracts
away a number of infrastructure issues, makes the code simpler, and is more
similar to how `TPUGANEstimator` must be run on cloud TPU.

The other estimator setup is exactly the same, but uses custom `tf.Estimator`
calls to train and evaluate.

#### Train and evaluate

<img src="images/tae_unconditional_gan.png" title="train_and_evaluate, unconditional GAN" width="330" />

#### Custom estimator calls

<img src="images/mnist_estimator_unconditional_gan.png" title="Custom calls, unconditional GAN" width="330" />
