## ESRGAN

### How to run
1. Run the setup instructions in [tensorflow_gan/examples/README.md](https://github.com/tensorflow/gan/blob/master/tensorflow_gan/examples/README.md#steps-to-run-an-example)
2. Run:
```
python esrgan/train.py
```

The Notebook files for training ESRGAN on Google Colaboratory can be found [here](colab_notebook)

### Description
The ESRGAN model proposed in the paper [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (Wang Xintao et al.)](https://arxiv.org/abs/1809.00219) performs the task of image super-resolution which is the process of reconstructing high resolution (HR) image from a given low resolution (LR) image. Here we have trained the ESRGAN model on the DIV2K dataset and the model is evaluated using TF-GAN.  

### Results
<img src="images/result1.png" title="Example 1" width="540" />
<img src="images/result2.png" title="Example 2" width="540" />
