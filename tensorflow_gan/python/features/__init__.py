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

"""TF-GAN features module.

This module includes support for virtual batch normalization, buffer replay,
conditioning, etc.
"""
# pylint: disable=wildcard-import,g-bad-import-order

# Collapse features into a single namespace.
from .clip_weights import *
from .conditioning_utils import *
from .normalization import *
from .random_tensor_pool import *
from .spectral_normalization import *
from .virtual_batchnorm import *

# Collect list of exposed symbols.
from .clip_weights import __all__ as clip_weights_symbols
from .conditioning_utils import __all__ as conditioning_utils_symbols
from .normalization import __all__ as normalization_symbols
from .random_tensor_pool import __all__ as random_tensor_pool_symbols
from .spectral_normalization import __all__ as spectral_normalization_symbols
from .virtual_batchnorm import __all__ as virtual_batchnorm_symbols

__all__ = clip_weights_symbols
__all__ += conditioning_utils_symbols
__all__ += normalization_symbols
__all__ += random_tensor_pool_symbols
__all__ += spectral_normalization_symbols
__all__ += virtual_batchnorm_symbols
