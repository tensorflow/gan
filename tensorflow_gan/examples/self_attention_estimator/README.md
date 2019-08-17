## GPU / TPU Self-Attention GAN Estimator on ImageNet

Authors: Yoel Drori, Augustus Odena, Joel Shor

### How to run


#### Run locally

1. Run the setup instructions in `tensorflow_gan/examples/README.md`
1. Run:

```shell
python self_attention_estimator/train_experiment_main.py
```

### Description

This code is a TF-GAN Estimator implementation of
[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318).
It can run locally, on GPU, and on cloud TPU.

Real images        | Generated images (GPU, 27 days)    | Generated images (TPU, 2 days)
----------------------------------------------------------------------------- | ----------- | -----------
<img src="images/imagegrid_real.png" title="Real images" width="300" /> | <img src="images/imagegrid_gpu.png" title="Generated images on GPU" width="300" /> | <img src="images/imagegrid_tpu.png" title="Generated images on TPU" width="300" />

Inception score and Frechet Inception distance based on step number:

<img src="images/tpu_vs_gpu_steps.png" title="Metrics by step" width="500" />

You can see that, as a function of train step, the GPU and TPU jobs are similar.
However, in terms of time, the TPU job is more than 12x faster:

<img src="images/tpu_vs_gpu_relative.png" title="Metrics by time" width="500" />
