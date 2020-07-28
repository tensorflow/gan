#!/bin/bash
#
# Some utilities for constructing, testing, and releasing PyPi packages.

make_virtual_env() {
  local py_version=$1
  local venv_dir=$2

  if [[ "$py_version" == "python2.7" ]]; then
    (exit 1) || echo "TF-GAN doesn't support Py2.X anymore".
  fi

  echo "make_virtual_env ${py_version} ${venv_dir}"

  # TODO(joelshor): Check that virtualenv exists and, if not, install it.
  # Probably check using something like `which virtualenv`, and install
  # using something like `sudo apt-get install virtualenv`.

  # Create and activate a virtualenv to specify python version and test in
  # isolated environment. Note that we don't actually have to cd'ed into a
  # virtualenv directory to use it; we just need to source bin/activate into the
  # current shell.
  VENV_PATH=${venv_dir}/virtualenv/${py_version}
  # Requires sudo apt-get install ${py_version}-venv. Seems not to work on
  # Debian with python3.6.
  eval ${py_version} -m venv ${VENV_PATH}
  source ${VENV_PATH}/bin/activate
}

install_tensorflow() {
  local tf_version=$1
  local py_version=$2

  if [[ "$py_version" == "python2.7" ]]; then
    (exit 1) || echo "TF-GAN doesn't support Py2.X anymore".
  fi

  if [[ "$tf_version" == "TF1.x" ]]; then
    pip3 install tensorflow==1.15
  elif [[ "$tf_version" == "TF2.x" ]]; then
    pip3 install tensorflow
  else
    echo "TensorFlow version not recognized: ${tf_version}"
    exit -1
  fi
}

install_tfp() {
  local tf_version=$1

  if [[ "$tf_version" == "TF1.x" ]]; then
    pip3 install tensorflow-probability==0.8.0
  elif [[ "$tf_version" == "TF2.x" ]]; then
    pip3 install tfp-nightly
  else
    echo "TensorFlow version not recognized: ${tf_version}"
    exit -1
  fi
}

install_tfds() {
  local tf_version=$1

  if [[ "$tf_version" == "TF1.x" ]]; then
    pip3 install tensorflow-datasets
  elif [[ "$tf_version" == "TF2.x" ]]; then
    pip3 install tfds-nightly
  else
    echo "TensorFlow version not recognized: ${tf_version}"
    exit -1
  fi
}

run_unittests_tests() {
  local py_version=$1
  local tf_version=$2

  if [[ "$py_version" == "python2.7" ]]; then
    (exit 1) || echo "TF-GAN doesn't support Py2.X anymore".
  fi
  echo "run_tests ${py_version}" "${tf_version}"
  venv_dir=$(mktemp -d)
  make_virtual_env "${py_version}" "${venv_dir}"

  # Install TensorFlow explicitly.
  install_tensorflow "${tf_version}" "${py_version}"

  # TODO(joelshor): These should get installed in setup.py, but aren't for some
  # reason.
  pip3 install Pillow
  pip3 install scipy
  pip3 install --upgrade google-api-python-client
  pip3 install --upgrade oauth2client
  install_tfp "${tf_version}"
  install_tfds "${tf_version}"
  pip3 install tensorflow-hub  # Package is the same regardless of TF version.

  # Run the tests.
  python setup.py test

  # Deactivate virtualenv.
  deactivate
}

test_build_and_install_whl() {
  local py_version=$1
  local tf_version=$2
  local venv_dir=$3

  if [[ "$py_version" == "python2.7" ]]; then
    (exit 1) || echo "TF-GAN doesn't support Py2.X anymore".
  fi

  echo "run_tests ${py_version}" "${tf_version}" "${venv_dir}"

  if [ "${venv_dir}" = "" ]; then
    venv_dir=$(mktemp -d)
  fi

  make_virtual_env "${py_version}" "${venv_dir}"

  pip3 install wheel
  # Install TensorFlow explicitly.
  install_tensorflow "${tf_version}"
  install_tfp "${tf_version}"
  pip3 install tensorflow-hub  # Package is the same regardless of TF version.

  # Install tf_gan package.
  WHEEL_PATH=${venv_dir}/wheel/${py_version}
  ./pip_pkg.sh ${WHEEL_PATH}/

  pip3 install ${WHEEL_PATH}/tensorflow_gan-*.whl

  # Move away from repo directory so "import tensorflow_gan" refers to the
  # installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import tensorflow_gan')

  # Deactivate virtualenv.
  deactivate
}

