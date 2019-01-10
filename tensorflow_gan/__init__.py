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

from tensorflow_gan.python import *  # pylint: disable=wildcard-import

# Cleanup symbols to avoid polluting namespace.
import sys as _sys  # pylint:disable=g-bad-import-order
delattr(_sys.modules[__name__], 'python')
delattr(_sys.modules[__name__], '_sys')
