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

"""TFGAN losses and penalties.

Losses can be used with individual arguments or with GANModel tuples.
"""
# pylint: disable=wildcard-import,g-bad-import-order

# Collapse losses into a single namespace.
from .tuple_losses import *
from tensorflow_gan.python.losses import losses_wargs as wargs

# Collect list of exposed symbols.
from tensorflow_gan.python.losses import tuple_losses
__all__ = ['wargs'] + tuple_losses.__all__
