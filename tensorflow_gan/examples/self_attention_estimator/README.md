## Self-Attention GAN Estimator on ImageNet

Authors: Yoel Drori, Augustus Odena, Joel Shor

### How to run


#### Run on cloud TPU <a id='cloud_tpu'></a>

1.  Set up your Cloud resources. This involves setting up a Cloud Bucket (disk),
    a TPU to run the computation, and Virtual Machine to run your code. There
    are multiple ways of bringing each of these up. The easiest is to follow the
    instructions in
    [this tutorial](https://cloud.google.com/tpu/docs/quickstart). Start at the
    top and finish after section `Verify your Compute Engine VM`.

    *   **Note**: Be sure to complete the `Clean Up` steps below after finishing
        to avoid incurring charges to your GCP account.
    *   **Note**: Be sure to read the
        [TPU pricing guide](https://cloud.google.com/tpu/pricing) and
        [Cloud Storage pricing guide](https://cloud.google.com/storage/pricing).
    *   **Note**: If you want to profile TPU utilization, be sure to install the
        [cloud tpu profiler](https://cloud.google.com/tpu/docs/cloud-tpu-tools#install_cloud_tpu_profiler).

1.  The final command should leave you connected to your new VM. If it hasn't,
    please follow
    [these instructions](https://cloud.google.com/compute/docs/instances/connecting-to-instance)
    to connect.

1.  On your new VM, run the following commands to download ImageNet from
    TensorFlow Datasets and convert them to TFRecords for easy training. This
    could take hours, so run it and take a coffee break:

    ```shell
    pip install --upgrade tensorflow_datasets --user
    tmux
    STORAGE_BUCKET=gs://YOUR-BUCKET-NAME
    python -c 'import tensorflow_datasets as tfds; ds = tfds.load("imagenet2012", split="train", data_dir="'${STORAGE_BUCKET}/data'"); tfds.as_numpy(ds)'
    ```

1.  Install the necessary packages and download the example code:

    ```shell
    git clone https://github.com/tensorflow/gan.git
    pip install tensorflow_gan --user
    ```

1.  Run the setup instructions in
    [tensorflow_gan/examples/README.md](https://github.com/tensorflow/gan/blob/master/tensorflow_gan/examples/README.md#steps-to-run-an-example)
    to properly set up the `PYTHONPATH`.

1.  Save the location of your cloud resources.

    ```shell
    export STORAGE_BUCKET=gs://YOUR-BUCKET-NAME
    export TPU_NAME=TPU-NAME
    export PROJECT_ID=PROJECT-ID
    export TPU_ZONE=ZONE
    ```

1.  Run the example:

    ```shell
    cd gan/tensorflow_gan/examples
    python self_attention_estimator/train_experiment_main.py \
      --use_tpu=true \
      --eval_on_tpu=true \
      --use_tpu_estimator=true \
      --mode=train_and_eval \
      --max_number_of_steps=999999 \
      --train_batch_size=1024 \
      --eval_batch_size=1024 \
      --num_eval_steps=49 \
      --train_steps_per_eval=1000 \
      --tpu=$TPU_NAME \
      --gcp_project=$PROJECT_ID \
      --tpu_zone=$TPU_ZONE \
      --model_dir=$STORAGE_BUCKET/logdir \
      --imagenet_data_dir=$STORAGE_BUCKET/data \
      --alsologtostderr
    ```

    *   **Note**: If you've run the data download step, training should start
        almost immediately. Otherwise, this will take a long time to start to
        run the first time, since the code needs to download the ImageNet
        dataset.
    *   **Note**: If your job starts downloading the data even though you ran
        the pre-download step, you probably didn't enter the same
        `STORAGE_BUCKET` location as in the previous step.
    *   **Note**: If your job fails with something like "Could not write to the
        internal temporary file.", you might need to follow
        [these instructions](https://cloud.google.com/tpu/docs/storage-buckets)
        and give the TPU permission to write to your cloud bucket.
    *   **Note**: If your job fails with "IOError: \[Errno 2\] No usable
        temporary directory found in ...", you might have run out of disk. Try
        clearing the temp directories listed and try again.
    * . **Note**: If your job fails with `Bad hardware status: ...`, try
        restarting your TPU.

1.  (Recommended) You can set up TensorBoard to track your training progress
    using
    [these instructions](https://cloud.google.com/tpu/docs/tensorboard-setup).

1.  Clean up by following the `Clean up` instructions in
    [this tutorial](https://cloud.google.com/tpu/docs/tutorials/efficientnet).

### Description

This code is a TF-GAN Estimator implementation of
[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318).
It can run locally, on GPU, and on cloud TPU.

Real images                                                                   | Generated images (GPU, 27 days)                                                          | Generated images (TPU, 2 days)
----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------
<img src="images/imagegrid_real.png" title="Real images" width="300" /> | <img src="images/imagegrid_gpu.png" title="Generated images on GPU" width="300" /> | <img src="images/imagegrid_tpu.png" title="Generated images on TPU" width="300" />

Inception score and Frechet Inception distance based on step number:

<img src="images/tpu_vs_gpu_steps.png" title="Metrics by step" width="500" />

You can see that, as a function of train step, the GPU and TPU jobs are similar.
However, in terms of time, the TPU job is more than 12x faster:

<img src="images/tpu_vs_gpu_relative.png" title="Metrics by time" width="500" />
