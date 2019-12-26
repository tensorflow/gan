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
"""Tests for tfgan.examples.networks.networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_gan.examples.cyclegan import networks


class NetworksTest(tf.test.TestCase):

    def test_generator_run(self):
        img_batch = tf.zeros([3, 128, 128, 3])
        model_output = networks.generator(img_batch)
        with self.cached_session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(model_output)

    def test_generator_graph(self):
        for shape in ([4, 32, 32], [3, 128, 128], [2, 80, 400]):
            tf.compat.v1.reset_default_graph()
            img = tf.ones(shape + [3])
            output_imgs = networks.generator(img)

            self.assertAllEqual(shape + [3], output_imgs.shape.as_list())

    def test_generator_graph_unknown_batch_dim(self):
        if tf.executing_eagerly():
            # Placeholders don't work in eager execution.
            return
        img = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 3])
        output_imgs = networks.generator(img)

        self.assertAllEqual([None, 32, 32, 3], output_imgs.shape.as_list())

    def test_generator_invalid_input(self):
        with self.assertRaisesRegexp(ValueError, 'must have rank 4'):
            networks.generator(tf.zeros([28, 28, 3]))

    def test_generator_run_multi_channel(self):
        img_batch = tf.zeros([3, 128, 128, 5])
        model_output = networks.generator(img_batch)
        with self.cached_session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(model_output)

    def test_generator_invalid_channels(self):
        if tf.executing_eagerly():
            # Placeholders don't work in eager execution.
            return
        with self.assertRaisesRegexp(
                ValueError, 'Last dimension shape must be known but is None'):
            img = tf.compat.v1.placeholder(tf.float32, shape=[4, 32, 32, None])
            networks.generator(img)

    def test_discriminator_run(self):
        img_batch = tf.zeros([3, 70, 70, 3])
        disc_output = networks.discriminator(img_batch)
        with self.cached_session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(disc_output)

    def test_discriminator_graph(self):
        # Check graph construction for a number of image size/depths and batch
        # sizes.
        for batch_size, patch_size in zip([3, 6], [70, 128]):
            tf.compat.v1.reset_default_graph()
            img = tf.ones([batch_size, patch_size, patch_size, 3])
            disc_output = networks.discriminator(img)

            self.assertEqual(2, disc_output.shape.ndims)
            self.assertEqual(batch_size, disc_output.shape.as_list()[0])

    def test_discriminator_invalid_input(self):
        try:
            networks.discriminator(tf.zeros([28, 28, 3]))
        except (ValueError, tf.errors.InvalidArgumentError):
            # TF raises ValueError, while TF2 raises tf.errors.InvalidArgumentError.
            return
        except Exception as e:  # pylint: disable=broad-except
            self.assertTrue(False, msg='Unexpected exception: {}'.format(e))
        self.assertTrue(False,
                        msg='Expected ValueError or InvalidArgumentError.')


if __name__ == '__main__':
    tf.test.main()
