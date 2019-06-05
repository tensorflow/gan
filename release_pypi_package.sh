#!/bin/bash
#
# A script for creating and releasing the TensorFlow GAN PyPi package.
#
# Usage:
#  ./release_pypi_package.sh {test|live}
#
# Order of operations of this script:
# 1) Build and test the whls (one for each desired version of Python).
# 2) Publish the whl (in test or live mode). Contact a TF-GAN team member if you
#    think you have a good reason to be building PyPi packages.
#
# To test that the package was uploaded, you can run the following:
#  $ pip install --index-url https://test.pypi.org/simple/ tensorflow-gan --upgrade
#  $ pip install tensorflow-gan --upgrade
#
# The resulting packages will be visible here:
#  $ https://pypi.org/project/tensorflow-gan/
#  $ https://test.pypi.org/project/tensorflow-gan/

set -e  # fail and exit on any command erroring

# Bring in useful functions.
source gbash.sh || exit 1
source module pypi_utils.sh

mode=$1

if [ "${mode}" = "live" ]; then
  TWINE_ARGS=""
elif [ "${mode}" = "test" ]; then
  TWINE_ARGS="--repository-url https://test.pypi.org/legacy/"
else
  echo "Need to pass 'live' or 'test' as first arg to script."
  exit 1
fi

echo "Building whl and test it..."
WHL_TMP_DIR=$(mktemp -d)
# TODO(joelshor): Add support for python3.x.
test_build_and_install_whl "python2.7" "TF1.x" "${WHL_TMP_DIR}"

# Publish to PyPI
read -p "Publish to ${mode}? (y/n) " -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
  make_virtual_env "python2.7" "TF1.x"
  # TODO(joelshor): Check that pip exists and, if not, install it.
  # Probably install using `sudo apt-get install python-pip`.

  # Check that twine exists. Note that on some systems, we must install it with
  # `pip install twine` and not `apt-get install twine`.
  pip install twine

  echo "Publishing to PyPI"
  # Since we must install using `pip install` (TODO(joelshor): Why?), we must
  # run twine inside python ex `python -m twine xxx` instead of `twine xxx`.
  # Note: Files should be of the form
  # ${WHL_TMP_DIR}/wheel/[python2.7|python3.5]/*.whl
  python -m twine upload ${TWINE_ARGS} ${WHL_TMP_DIR}/wheel/*/*

  # Deactivate virtualenv.
  deactivate
else
  echo "Skipping upload"
  exit 1
fi

rm -rf $TMP_DIR1
rm -rf $TMP_DIR2
