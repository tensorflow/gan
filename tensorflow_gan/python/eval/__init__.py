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

"""TF-GAN evaluation module.

This module supports techniques such as Inception Score, Frechet Inception
distance, and Sliced Wasserstein distance.
"""
# pylint: disable=wildcard-import,g-bad-import-order

# Collapse eval into a single namespace.
from .classifier_metrics import *
from .eval_utils import *
from .inception_metrics import *
from .sliced_wasserstein import *
from .summaries import *

# Collect list of exposed symbols.
from .classifier_metrics import __all__ as classifier_metrics_symbols
from .eval_utils import __all__ as eval_utils_symbols
from .inception_metrics import __all__ as inception_metrics_symbols
from .sliced_wasserstein import __all__ as sliced_wasserstein_symbols
from .summaries import __all__ as summaries_symbols
__all__ = classifier_metrics_symbols
__all__ += eval_utils_symbols
__all__ += inception_metrics_symbols
__all__ += sliced_wasserstein_symbols
__all__ += summaries_symbols
