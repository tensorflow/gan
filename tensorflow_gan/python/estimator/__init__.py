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

"""TF-GAN estimator module.

GANEstimator provides all the infrastructure support of a TensorFlow Estimator
with the feature support of TF-GAN.
"""
# pylint: disable=wildcard-import,g-bad-import-order

# Collapse `estimator` into a single namespace.
from .gan_estimator import *
from .latent_gan_estimator import *
from .stargan_estimator import *
from .tpu_gan_estimator import *

# Collect list of exposed symbols.
from .gan_estimator import __all__ as gan_estimator_symbols
from .latent_gan_estimator import __all__ as latent_gan_estimator_symbols
from .stargan_estimator import __all__ as stargan_estimator_symbols
from .tpu_gan_estimator import __all__ as tpu_gan_estimator_symbols
__all__ = gan_estimator_symbols
__all__ += latent_gan_estimator_symbols
__all__ += stargan_estimator_symbols
__all__ += tpu_gan_estimator_symbols
