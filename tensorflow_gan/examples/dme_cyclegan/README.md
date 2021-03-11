## DME CycleGAN

A variant of cycleGAN used in the paper ["Scientific Discovery by Generating Counterfactuals using
Image Translation"](https://arxiv.org/abs/2007.05500), MICCAI, 2020.

Authors: Subhashini Venuopalan, Arunachalam Narayanaswamy

### How to run


1.  Run the setup instructions in [tensorflow_gan/examples/README.md](https://github.com/tensorflow/gan/blob/master/tensorflow_gan/examples/README.md#steps-to-run-an-example)
1.  Install `pillow` with ex `pip install Pillow`.
1.  Download the
    [`tensorflow_models` github repo](https://github.com/tensorflow/models) for
    access to network architectures ex `git clone
    https://github.com/tensorflow/models.git $TF_MODELS_DIR`.
1.  Add the 'research' sub-directory to the PYTHONPATH with ex: `export
    PYTHONPATH=${TF_MODELS_DIR}/research:${PYTHONPATH}`
1.  Run:

```
python dme_cyclegan/train.py
```


