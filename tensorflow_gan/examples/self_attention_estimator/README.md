## Self-Attention GAN Estimator on ImageNet

Authors: Yoel Drori, Augustus Odena, Joel Shor

### How to run


1.  Run the setup instructions in `tensorflow_gan/examples/README.md`
1.  Run:

```shell
python self_attention_estimator/train_experiment.py
```

### Description

This code is a TF-GAN Estimator implementation of
[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318).

Generated images                                                              | Real images
----------------------------------------------------------------------------- | -----------
<img src="images/imagegrid.png" title="Generated images" width="300" /> | <img src="images/imagegrid_real.png" title="Real images" width="300" />
