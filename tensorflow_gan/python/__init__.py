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
# ============================================================================
# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TF-GAN is a lightweight library for training and evaluating GANs.

In addition to providing the infrastructure for easily training and evaluating
GANS, this library contains modules for a TF-GAN-backed Estimator,
evaluation metrics, features (such as virtual batch normalization), and losses.
Please see README.md for details and usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Collapse TF-GAN into a tiered namespace.
# Module names to keep.
from tensorflow_gan.python import estimator
from tensorflow_gan.python import eval  # pylint:disable=redefined-builtin
from tensorflow_gan.python import features
from tensorflow_gan.python import losses

# Modules to wildcard import.
from tensorflow_gan.python import namedtuples
from tensorflow_gan.python import train
from tensorflow_gan.python.namedtuples import *  # pylint:disable=wildcard-import
from tensorflow_gan.python.train import *  # pylint:disable=wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

allowed_symbols = [
    'estimator',
    'eval',
    'features',
    'losses',
]
allowed_symbols += namedtuples.__all__
allowed_symbols += train.__all__

remove_undocumented(estimator.__name__, estimator.allowed_symbols)
remove_undocumented(eval.__name__, eval.allowed_symbols)
remove_undocumented(features.__name__, features.allowed_symbols)
remove_undocumented(losses.__name__, losses.allowed_symbols)
remove_undocumented(__name__, allowed_symbols)
