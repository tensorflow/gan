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

  # Enable python3.6 in pyenv and update its pip.
  
  # Log versions to debug
  pyenv versions
  pyenv install --list

  # By default Ubuntu 16.04 uses Python 3.5.
  pyenv install 3.6.8
  pyenv global 3.6.8
  which python
  python --version

  # Create and activate a virtualenv to specify python version and test in
  # isolated environment. Note that we don't actually have to cd'ed into a
  # virtualenv directory to use it; we just need to source bin/activate into the
  # current shell.
  VENV_PATH=${venv_dir}/virtualenv/${py_version}
  virtualenv -p "${py_version}" "${VENV_PATH}"
  source ${VENV_PATH}/bin/activate
}

install_tensorflow() {
  local tf_version=$1
  local py_version=$2

  if [[ "$py_version" == "python2.7" ]]; then
    (exit 1) || echo "TF-GAN doesn't support Py2.X anymore".
  fi

  # Workaround for https://github.com/tensorflow/tensorflow/issues/32319.
  pip install gast==0.2.2

  if [[ "$tf_version" == "TF1.x" ]]; then
    pip install tensorflow==1.15
  elif [[ "$tf_version" == "TF2.x" ]]; then
    pip install tensorflow
  else
    echo "TensorFlow version not recognized: ${tf_version}"
    exit -1
  fi
}

install_tfp() {
  local tf_version=$1

  if [[ "$tf_version" == "TF1.x" ]]; then
    pip install tensorflow-probability==0.8.0
  elif [[ "$tf_version" == "TF2.x" ]]; then
    pip install tfp-nightly
  else
    echo "TensorFlow version not recognized: ${tf_version}"
    exit -1
  fi
}

install_tfds() {
  local tf_version=$1

  if [[ "$tf_version" == "TF1.x" ]]; then
    pip install tensorflow-datasets
  elif [[ "$tf_version" == "TF2.x" ]]; then
    pip install tfds-nightly
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
  pip install Pillow
  pip install scipy
  pip install --upgrade google-api-python-client
  pip install --upgrade oauth2client
  install_tfp "${tf_version}"
  install_tfds "${tf_version}"
  pip install tensorflow-hub  # Package is the same regardless of TF version.

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

  # Install TensorFlow explicitly.
  install_tensorflow "${tf_version}"
  install_tfp "${tf_version}"
  pip install tensorflow-hub  # Package is the same regardless of TF version.

  # Install tf_gan package.
  WHEEL_PATH=${venv_dir}/wheel/${py_version}
  ./pip_pkg.sh ${WHEEL_PATH}/

  pip install ${WHEEL_PATH}/tensorflow_gan-*.whl

  # Move away from repo directory so "import tensorflow_gan" refers to the
  # installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import tensorflow_gan')

  # Deactivate virtualenv.
  deactivate
}

