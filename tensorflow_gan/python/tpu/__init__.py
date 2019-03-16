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

"""TF-GAN TPU support."""
# pylint: disable=wildcard-import,g-bad-import-order,line-too-long,g-no-space-after-comment

# Collapse losses into a single namespace.
# TODO(joelshor): Figure out why including this line breaks the open source
# build
# from tensorflow_gan.python.tpu.cross_replica_ops import *
from tensorflow_gan.python.tpu.normalization_ops import *

# Collect list of exposed symbols.
#from tensorflow_gan.python.tpu.cross_replica_ops import __all__ as cross_replica_ops_symbols
from tensorflow_gan.python.tpu.normalization_ops import __all__ as normalization_ops_symbols
__all__ = normalization_ops_symbols
# __all__ += cross_replica_ops_symbols
