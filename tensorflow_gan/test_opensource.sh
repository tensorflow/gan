#!/bin/bash

YOUR_LDAP=cassandrax
YOUR_CITC_CLIENT=gan
YOUR_CL_NUM=179232106

# Script to automate parts of going from piper to github.
# Test runs the generated opensource code.

# 0) Install TF from nightly build
# Note, you need to be on Rodete or else this will fail b/73389237.
pip install tf-nightly --user

# 1) Download the tensorflow models repo
cd ~
git clone https://github.com/tensorflow/gan.git

# 2) Change branches in the local git repo
cd ~/models
git checkout -b tmp

# 3) Change the url variable in the piper_to_local workflow to be the path of
#    the downloaded repo. A typical value if you followed step 1 is:
#    file:///usr/local/google/home/$YOUR_LDAP/models
# 4) From the root of your citc client, run the following command. If testing
#    a local client change, add the CL number at the end.

URL_PLACEHOLDER='url = local_directory'
URL_VARIABLE='url = "file:\/\/\/usr\/local\/google\/home\/$YOUR_LDAP\/models"'

g4d $YOUR_CITC_CLIENT && \
sed -i 's/'"$URL_PLACEHOLDER"'/'"$URL_VARIABLE"'/g' \
  third_party/tensorflow_models/gan/copy.bara.sky && \
/google/data/ro/teams/copybara/copybara \
  third_party/tensorflow_models/gan/copy.bara.sky piper_to_local \
  $YOUR_CL_NUM --force &&\
sed -i 's/'"$URL_VARIABLE"'/'"$URL_PLACEHOLDER"'/g' \
third_party/tensorflow_models/gan/copy.bara.sky

# 5) Go back to to your checked out git repo. The master branch will have your
#    change as a new commit.
cd ~/models
git checkout master
git diff HEAD~1

# Add tests per module. Currently only "cyclegan" subdir is tested here.

# Module: cyclegan
# Set the Python path so that modules can be successfully imported.
GIT_REPO=/usr/local/google/home/$YOUR_LDAP/models
export PYTHONPATH=$PYTHONPATH:$GIT_REPO:$GIT_REPO/research:$GIT_REPO/research/gan/cyclegan:$GIT_REPO/research/gan/pix2pix:$GIT_REPO/research/gan
TESTDATA_GLOB=/google/src/cloud/cassandrax/gan/google3/third_party/tensorflow_models/gan/cyclegan/testdata/*.jpg
LOGDIR=/tmp/cyclegan/

cd $GIT_REPO/research/gan/cyclegan
rm -R $LOGDIR
python train.py \
  --image_set_x_file_pattern=$TESTDATA_GLOB \
  --image_set_y_file_pattern=$TESTDATA_GLOB \
  --patch_size=8 \
  --train_logdir=$LOGDIR

# Get the first checkpoint.
CHECKPOINTS=( $LOGDIR/*.meta )
CHECKPOINT=${CHECKPOINTS[0]}
CHECKPOINT=${CHECKPOINT%.meta}

python inference_demo.py \
  --checkpoint_path=$CHECKPOINT \
  --image_set_x_glob=$TESTDATA_GLOB \
  --image_set_y_glob=$TESTDATA_GLOB
