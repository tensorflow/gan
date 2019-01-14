#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the CIFAR dataset.
# 2. Trains an unconditional model on the CIFAR training set.
# 3. Evaluates the models and writes sample images to disk.
#
#
# With the default batch size and number of steps, train times are:
#
# Usage:
# cd models/research/gan/cifar
# ./launch_jobs.sh ${git_repo}
set -e

# Location of the git repository.
git_repo=$2
if [[ "$git_repo" == "" ]]; then
  echo "'git_repo' must not be empty."
  exit
fi

# Base name for where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/cifar-model

# Base name for where the evaluation images will be saved to.
EVAL_DIR=/tmp/cifar-model/eval

# Where the dataset is saved to.
DATASET_DIR=/tmp/cifar-data

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
  --dataset_name=cifar10 \
  --dataset_dir=${DATASET_DIR}

# Run unconditional GAN.
UNCONDITIONAL_TRAIN_DIR="${TRAIN_DIR}/unconditional"
UNCONDITIONAL_EVAL_DIR="${EVAL_DIR}/unconditional"
NUM_STEPS=10000
# Run training.
Banner "Starting training unconditional GAN for ${NUM_STEPS} steps..."
python "${git_repo}/research/gan/cifar/train.py" \
  --train_log_dir=${UNCONDITIONAL_TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --max_number_of_steps=${NUM_STEPS} \
  --alsologtostderr
Banner "Finished training unconditional GAN ${NUM_STEPS} steps."

# Run evaluation.
Banner "Starting evaluation of unconditional GAN..."
python "${git_repo}/research/gan/cifar/eval.py" \
  --checkpoint_dir=${UNCONDITIONAL_TRAIN_DIR} \
  --eval_dir=${UNCONDITIONAL_EVAL_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --eval_real_images=false \
  --conditional_eval=false \
  --max_number_of_evaluations=1
Banner "Finished unconditional evaluation. See ${UNCONDITIONAL_EVAL_DIR} for output images."
