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

"""Tests for CycleGAN inference demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import logging
import numpy as np
import PIL

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.cyclegan import inference_demo
from tensorflow_gan.examples.cyclegan import train_lib

FLAGS = flags.FLAGS
mock = tf.compat.v1.test.mock


def _basenames_from_glob(file_glob):
  return [
      os.path.basename(file_path) for file_path in tf.io.gfile.glob(file_glob)
  ]


class InferenceDemoTest(tf.test.TestCase):

  def setUp(self):
    self._export_dir = os.path.join(FLAGS.test_tmpdir, 'export')
    self._ckpt_path = os.path.join(self._export_dir, 'model.ckpt')
    self._image_glob = os.path.join(
        FLAGS.test_srcdir,
        'tensorflow_gan/examples/cyclegan/testdata',
        '*.jpg')
    self._genx_dir = os.path.join(FLAGS.test_tmpdir, 'genx')
    self._geny_dir = os.path.join(FLAGS.test_tmpdir, 'geny')

  @mock.patch.object(tfgan, 'gan_train', autospec=True)
  @mock.patch.object(
      train_lib.data_provider, 'provide_custom_data', autospec=True)
  def testTrainingAndInferenceGraphsAreCompatible(
      self, mock_provide_custom_data, unused_mock_gan_train):
    if tf.executing_eagerly():
      # `tfgan.cyclegan_model` doesn't work when executing eagerly.
      return
    # Training and inference graphs can get out of sync if changes are made
    # to one but not the other. This test will keep them in sync.

    # Save the training graph
    train_sess = tf.compat.v1.Session()
    hparams = train_lib.HParams(
        image_set_x_file_pattern='/tmp/x/*.jpg',
        image_set_y_file_pattern='/tmp/y/*.jpg',
        batch_size=3,
        patch_size=128,
        master='master',
        train_log_dir=self._export_dir,
        generator_lr=0.02,
        discriminator_lr=0.3,
        max_number_of_steps=1,
        ps_replicas=0,
        task=0,
        cycle_consistency_loss_weight=2.0)
    mock_provide_custom_data.return_value = (
        tf.zeros([3, 4, 4, 3,]), tf.zeros([3, 4, 4, 3]))
    train_lib.train(hparams)
    init_op = tf.compat.v1.global_variables_initializer()
    train_sess.run(init_op)
    train_saver = tf.compat.v1.train.Saver()
    train_saver.save(train_sess, save_path=self._ckpt_path)

    # Create inference graph
    tf.compat.v1.reset_default_graph()
    FLAGS.patch_dim = hparams.patch_size
    logging.info('dir_path: %s', os.listdir(self._export_dir))
    FLAGS.checkpoint_path = self._ckpt_path
    FLAGS.image_set_x_glob = self._image_glob
    FLAGS.image_set_y_glob = self._image_glob
    FLAGS.generated_x_dir = self._genx_dir
    FLAGS.generated_y_dir = self._geny_dir

    inference_demo.main(None)
    logging.info('gen x: %s', os.listdir(self._genx_dir))

    # Check that the image names match
    self.assertSetEqual(
        set(_basenames_from_glob(FLAGS.image_set_x_glob)),
        set(os.listdir(FLAGS.generated_y_dir)))
    self.assertSetEqual(
        set(_basenames_from_glob(FLAGS.image_set_y_glob)),
        set(os.listdir(FLAGS.generated_x_dir)))

    # Check that each image in the directory looks as expected
    for directory in [FLAGS.generated_x_dir, FLAGS.generated_x_dir]:
      for base_name in os.listdir(directory):
        image_path = os.path.join(directory, base_name)
        self.assertRealisticImage(image_path)

  def assertRealisticImage(self, image_path):
    logging.info('Testing %s for realism.', image_path)
    # If the normalization is off or forgotten, then the generated image is
    # all one pixel value. This tests that different pixel values are achieved.
    input_np = np.asarray(PIL.Image.open(image_path))
    self.assertEqual(len(input_np.shape), 3)
    self.assertGreaterEqual(input_np.shape[0], 50)
    self.assertGreaterEqual(input_np.shape[1], 50)
    self.assertGreater(np.mean(input_np), 20)
    self.assertGreater(np.var(input_np), 100)


if __name__ == '__main__':
  tf.test.main()
