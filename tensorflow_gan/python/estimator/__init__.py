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
# Copyright 2016 Google Inc. All Rights Reserved.
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
"""TF-GAN estimator module.

GANEstimator provides all the infrastructure support of a TensorFlow Estimator
with the feature support of TF-GAN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Collapse `estimator` into a single namespace.
from tensorflow_gan.python.estimator import gan_estimator
from tensorflow_gan.python.estimator import latent_gan_estimator
from tensorflow_gan.python.estimator import stargan_estimator
from tensorflow_gan.python.estimator import tpu_gan_estimator

# pylint: disable=wildcard-import
from tensorflow_gan.python.estimator.gan_estimator import *
from tensorflow_gan.python.estimator.latent_gan_estimator import *
from tensorflow_gan.python.estimator.stargan_estimator import *
from tensorflow_gan.python.estimator.tpu_gan_estimator import *
# pylint: enable=wildcard-import

allowed_symbols = gan_estimator.__all__
allowed_symbols += latent_gan_estimator.__all__
allowed_symbols += stargan_estimator.__all__
allowed_symbols += tpu_gan_estimator.__all__
