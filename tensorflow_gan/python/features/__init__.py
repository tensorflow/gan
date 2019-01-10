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
# Copyright 2017 Google Inc. All Rights Reserved.
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
"""TF-GAN features module.

This module includes support for virtual batch normalization, buffer replay,
conditioning, etc.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Collapse features into a single namespace.
from tensorflow_gan.python.features import clip_weights
from tensorflow_gan.python.features import conditioning_utils
from tensorflow_gan.python.features import random_tensor_pool
from tensorflow_gan.python.features import spectral_normalization
from tensorflow_gan.python.features import virtual_batchnorm

# pylint: disable=unused-import,wildcard-import
from tensorflow_gan.python.features.clip_weights import *
from tensorflow_gan.python.features.conditioning_utils import *
from tensorflow_gan.python.features.random_tensor_pool import *
from tensorflow_gan.python.features.spectral_normalization import *
from tensorflow_gan.python.features.virtual_batchnorm import *
# pylint: enable=unused-import,wildcard-import

allowed_symbols = clip_weights.__all__
allowed_symbols += conditioning_utils.__all__
allowed_symbols += random_tensor_pool.__all__
allowed_symbols += spectral_normalization.__all__
allowed_symbols += virtual_batchnorm.__all__
