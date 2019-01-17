# coding=utf-8
# Copyright 2018 The TensorFlow GAN Authors.
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

"""Install TF-GAN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import io
import os
import sys
import unittest

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.command.test import test as TestCommandBase
from setuptools.dist import Distribution

from tensorflow_gan import __version__  # pylint: disable=g-import-not-at-top

project_name = 'tensorflow-gan'
version = __version__


class StderrWrapper(io.IOBase):

  def write(self, *args, **kwargs):
    return sys.stderr.write(*args, **kwargs)

  def writeln(self, *args, **kwargs):
    if args or kwargs:
      sys.stderr.write(*args, **kwargs)
    sys.stderr.write('\n')


class Test(TestCommandBase):

  def run_tests(self):
    # Import absl inside run, where dependencies have been loaded already.
    from absl import app  # pylint: disable=g-import-not-at-top

    def main(_):
      test_loader = unittest.TestLoader()
      test_suite = test_loader.discover('tensorflow_gan', pattern='*_test.py')
      stderr = StderrWrapper()
      result = unittest.TextTestResult(stderr, descriptions=True, verbosity=2)
      test_suite.run(result)

      result.printErrors()

      final_output = ('Tests run: {}. Errors: {}  Failures: {}.'.format(
          result.testsRun, len(result.errors), len(result.failures)))

      header = '=' * len(final_output)
      stderr.writeln(header)
      stderr.writeln(final_output)
      stderr.writeln(header)

      if result.wasSuccessful():
        return 0
      else:
        return 1

    # Run inside absl.app.run to ensure flags parsing is done.
    return app.run(main)


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(
    name=project_name,
    version=version,
    description=('TF-GAN:A Generative Adversarial Networks Library for '
                 'TensorFlow'),
    long_description=long_description,
    author='Joel Shor',
    author_email='joelshor@google.com',
    url='http://github.com/tensorflow/gan',
    license='Apache 2.0',
    packages=find_packages(),
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'test': Test,
        'pip_pkg': InstallCommandBase,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow GAN generative model machine learning',
)
