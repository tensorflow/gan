#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the MNIST dataset.
# 2. Trains an unconditional model on the MNIST training set using a
#    tf.Estimator.
# 3. Evaluates the models and writes sample images to disk.
#
#
# Usage:
# cd tensorflow_gan/examples/mnist_estimator
# ./launch_jobs.sh ${git_repo}
set -e

# Location of the git repository.
git_repo=$1
if [[ "$git_repo" == "" ]]; then
  echo "'git_repo' must not be empty."
  exit
fi

# Base name for where the evaluation images will be saved to.
output_dir=/tmp/mnist-estimator

# Where the dataset is saved to.
DATASET_DIR=/tmp/mnist-data

export PYTHONPATH=$PYTHONPATH:$git_repo:$git_repo/research:$git_repo/research/gan:$git_repo/research/slim

# A helper function for printing pretty output.
Banner () {
  local text=$1
  local green='\033[0;32m'
  local nc='\033[0m'  # No color.
  echo -e "${green}${text}${nc}"
}

# Download the dataset.
python "${git_repo}/research/slim/download_and_convert_data.py" \
  --dataset_name=mnist \
  --dataset_dir=${DATASET_DIR}

# Run unconditional GAN.
NUM_STEPS=1600
Banner "Starting training GANEstimator ${NUM_STEPS} steps..."
python "${git_repo}/tensorflow_gan/examples/mnist_estimator/train.py" \
  --max_number_of_steps=${NUM_STEPS} \
  --output_dir=${OUTPUT_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --alsologtostderr
Banner "Finished training GANEstimator ${NUM_STEPS} steps. See ${OUTPUT_DIR} for output images."
