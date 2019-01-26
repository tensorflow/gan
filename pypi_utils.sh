#!/bin/bash
#
# Some utilities for constructing, testing, and releasing PyPi packages.

make_virtual_env() {
  local py_version=$1
  local venv_dir=$2
  echo "make_virtual_env ${py_version} ${venv_dir}"

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

  if [[ "$tf_version" == "TF1.x" ]]; then
    pip install tensorflow
  elif [[ "$tf_version" == "TF2.x" ]]; then
    pip install tf-nightly-2.0-preview
  else
    echo "TensorFlow version not recognized: ${tf_version}"
    exit -1
  fi
}

run_unittests_tests() {
  local py_version=$1
  local tf_version=$2
  echo "run_tests ${py_version}" "${tf_version}"
  venv_dir=$(mktemp -d)
  make_virtual_env "${py_version}" "${venv_dir}"

  # Install TensorFlow explicitly (see http://g/tf-oss/vpEioAGbZ4Q).
  install_tensorflow "${tf_version}"

  # TODO(joelshor): These should get installed in setup.py, but aren't for some
  # reason.
  pip install scipy
  pip install tensorflow-probability

  # Run the tests.
  python setup.py test

  # Deactivate virtualenv
  deactivate
}

test_build_and_install_whl() {
  local py_version=$1
  local tf_version=$2
  local venv_dir=$3

  echo "run_tests ${py_version}" "${tf_version}" "${venv_dir}"

  if [ "${venv_dir}" = "" ]; then
    venv_dir=$(mktemp -d)
  fi

  make_virtual_env "${py_version}" "${venv_dir}"

  # Install TensorFlow explicitly (see http://g/tf-oss/vpEioAGbZ4Q).
  install_tensorflow "${tf_version}"

  # Install tf_gan package.
  WHEEL_PATH=${venv_dir}/wheel/${py_version}
  ./pip_pkg.sh ${WHEEL_PATH}/

  pip install ${WHEEL_PATH}/tensorflow_gan-*.whl

  # Move away from repo directory so "import tensorflow_gan" refers to the
  # installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import tensorflow_gan')

  # Deactivate virtualenv
  deactivate
}

