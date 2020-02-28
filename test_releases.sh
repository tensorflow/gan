#!/bin/bash

# Exit if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x

# Bring in useful functions.
source "pypi_utils.sh"

# Run script.

if ! which virtualenv > /dev/null; then
   echo -e "virtualenv not found! needed for tests. Install? (y/n)"
   read REPLY
   if  [ "$REPLY" = "y" ]; then
      sudo apt-get install virtualenv
   fi
fi
# Run unit tests.
run_unittests_tests "python2.7" "TF1.x"
run_unittests_tests "python2.7" "TF2.x"
run_unittests_tests "python3.8" "TF1.x"
run_unittests_tests "python3.8" "TF2.x"

# Test that we can build the whl.
test_build_and_install_whl "python2.7" "TF1.x"
test_build_and_install_whl "python2.7" "TF2.x"
test_build_and_install_whl "python3.8" "TF1.x"
test_build_and_install_whl "python3.8" "TF2.x"

