# coding=utf-8
# Copyright 2019 The TensorFlow GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TF-GAN is a lightweight library for training and evaluating GANs.

In addition to providing the infrastructure for easily training and evaluating
GANS, this library contains modules for a TF-GAN-backed Estimator,
evaluation metrics, features (such as virtual batch normalization), and losses.
Please see README.md for details and usage.
"""
# We need to put some imports inside a function call below, and the function
# call needs to come before the *actual* imports that populate the
# tensorflow_gan namespace. Hence, we disable this lint check throughout
# the file.
#
# pylint: disable=g-import-not-at-top


# Ensure TensorFlow is importable and its version is sufficiently recent. This
# needs to happen before anything else, since the imports below will try to
# import tensorflow, too.
def _ensure_tf_install():  # pylint: disable=g-statement-before-imports
  """Attempt to import tensorflow, and ensure its version is sufficient.

  Raises:
    ImportError: if either tensorflow is not importable or its version is
    inadequate.
  """
  try:
    import tensorflow as tf
  except ImportError:
    # Print more informative error message, then reraise.
    print("\n\nFailed to import TensorFlow. Please note that TensorFlow is not "
          "installed by default when you install TF-GAN. This is so that users "
          "can decide whether to install the GPU-enabled TensorFlow package. "
          "To use TF-GAN, please install the most recent version of "
          "TensorFlow, by following instructions at "
          "https://tensorflow.org/install.\n\n")
    raise

  import distutils.version

  #
  # Update this whenever we need to depend on a newer TensorFlow release.
  #
  required_tensorflow_version = "1.14.0"

  if (distutils.version.LooseVersion(tf.__version__) <
      distutils.version.LooseVersion(required_tensorflow_version)):
    raise ImportError(
        "This version of TF-GAN requires TensorFlow "
        "version >= {required}; Detected an installation of version {present}. "
        "Please upgrade TensorFlow to proceed.".format(
            required=required_tensorflow_version,
            present=tf.__version__))


# Ensure TensorFlow Probability is importable and its version is sufficiently
# recent. This needs to happen before anything else, since the imports below
# will try to import tensorflow_probability, too.
def _ensure_tfp_install():  # pylint: disable=g-statement-before-imports
  """Attempt to import tensorflow, and ensure its version is sufficient.

  Raises:
    ImportError: if either tensorflow_probability is not importable or its
    version is inadequate.
  """
  try:
    import tensorflow_probability as tfp
  except ImportError:
    # Print more informative error message, then reraise.
    print("\n\nFailed to import TensorFlow Probability. "
          "To use TF-GAN, please install the most recent version of "
          "TensorFlow Probability, by following instructions at "
          "https://www.tensorflow.org/probability/install.\n\n")
    raise

  import distutils.version

  #
  # Update this whenever we need to depend on a newer TensorFlow Probability
  # release.
  #
  required_tensorflow_probability_version = "0.7"

  if (distutils.version.LooseVersion(tfp.__version__) <
      distutils.version.LooseVersion(required_tensorflow_probability_version)):
    raise ImportError(
        "This version of TF-GAN requires TensorFlow Probability "
        "version >= {required}; Detected an installation of version {present}. "
        "Please upgrade TensorFlow to proceed.".format(
            required=required_tensorflow_probability_version,
            present=tfp.__version__))


_ensure_tf_install()
_ensure_tfp_install()

from tensorflow_gan.python import *  # pylint: disable=wildcard-import

# Cleanup symbols to avoid polluting namespace.
import sys as _sys  # pylint:disable=g-bad-import-order
delattr(_sys.modules[__name__], "python")
delattr(_sys.modules[__name__], "_ensure_tf_install")
delattr(_sys.modules[__name__], "_ensure_tfp_install")
delattr(_sys.modules[__name__], "_sys")
