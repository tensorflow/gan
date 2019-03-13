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

We construct the interface here, and remove undocumented symbols. The high-level
structure should be:
tfgan
-> .estimator
-> .eval
-> .features
-> .losses
  -> .wargs
-> .tpu
"""
# pylint:disable=g-import-not-at-top,g-bad-import-order

# Collapse TF-GAN into a tiered namespace.
# Module names to keep.
from tensorflow_gan.python import estimator
from tensorflow_gan.python import eval  # pylint:disable=redefined-builtin
from tensorflow_gan.python import features
from tensorflow_gan.python import losses
from tensorflow_gan.python import tpu

# Modules to wildcard import.
from tensorflow_gan.python.namedtuples import *  # pylint:disable=wildcard-import
from tensorflow_gan.python.train import *  # pylint:disable=wildcard-import

# Get the version number.
from tensorflow_gan.python.version import __version__

# Collect allowed top-level symbols to expose to users.
__all__ = [
    'estimator',
    'eval',
    'features',
    'losses',
    'tpu',
    '__version__',
]
from tensorflow_gan.python.namedtuples import __all__ as namedtuple_symbols
from tensorflow_gan.python.train import __all__ as train_symbols
__all__ += namedtuple_symbols
__all__ += train_symbols

# Remove undocumented symbols to avoid polluting namespaces.
