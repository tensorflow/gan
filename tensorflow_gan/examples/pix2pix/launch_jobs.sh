#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Imagenet dataset.
# 2. Trains image compression model on patches from Imagenet.
# 3. Evaluates the models and writes sample images to disk.
#
# Usage:
# cd models/research/gan/image_compression
# ./launch_jobs.sh ${weight_factor} ${git_repo}
set -e

# Weight of the adversarial loss.
weight_factor=$1
if [[ "$weight_factor" == "" ]]; then
  echo "'weight_factor' must not be empty."
  exit
fi

# Location of the git repository.
git_repo=$2
if [[ "$git_repo" == "" ]]; then
  echo "'git_repo' must not be empty."
  exit
fi

# Base name for where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/compression-model

# Base name for where the evaluation images will be saved to.
EVAL_DIR=/tmp/compression-model/eval

# Where the dataset is saved to.
DATASET_DIR=/tmp/imagenet-data

export PYTHONPATH=$PYTHONPATH:$git_repo:$git_repo/research:$git_repo/research/slim:$git_repo/research/slim/nets

# A helper function for printing pretty output.
Banner () {
  local text=$1
  local green='\033[0;32m'
  local nc='\033[0m'  # No color.
  echo -e "${green}${text}${nc}"
}

# Download the dataset.
bazel build "${git_repo}/research/slim:download_and_convert_imagenet"
"./bazel-bin/download_and_convert_imagenet" ${DATASET_DIR}

# Run the pix2pix model.
NUM_STEPS=10000
MODEL_TRAIN_DIR="${TRAIN_DIR}/wt${weight_factor}"
Banner "Starting training an image compression model for ${NUM_STEPS} steps..."
python "${git_repo}/research/gan/image_compression/train.py" \
  --train_log_dir=${MODEL_TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --max_number_of_steps=${NUM_STEPS} \
  --weight_factor=${weight_factor} \
  --alsologtostderr
Banner "Finished training pix2pix model ${NUM_STEPS} steps."
