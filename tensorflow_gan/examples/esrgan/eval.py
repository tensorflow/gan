# coding=utf-8
# Copyright 2021 The TensorFlow GAN Authors.
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

from absl import flags, logging, app
import tensorflow as tf
import eval_lib
import data_provider


flags.DEFINE_integer('batch_size', 2,
                     'The number of images in each batch.')
flags.DEFINE_integer('hr_dimension', 128,
                     'Dimension of a HR image.')
flags.DEFINE_integer('scale', 4,
                     'Factor by which LR images are downscaled.')                     
flags.DEFINE_string('model_dir', '/content/',
                    'Directory where the trained models are stored.')
flags.DEFINE_string('data_dir', '/content/datasets',
                    'Directory where dataset is stored.')
flags.DEFINE_integer('num_steps', 2,
                     'The number of steps for evaluation.')
flags.DEFINE_integer('num_inception_images', 2,
                     'The number of images passed for evaluation at each step.')
flags.DEFINE_string('image_dir', '/content/results',
                    'Directory to save generated images during evaluation.')
flags.DEFINE_boolean('eval_real_images', False,
                     'Whether Phase 1 training is done or not')

FLAGS = flags.FLAGS

def main(_):
  hparams = eval_lib.HParams(FLAGS.batch_size, FLAGS.hr_dimension, 
                             FLAGS.scale, FLAGS.model_dir, 
                             FLAGS.data_dir,FLAGS.num_steps, 
                             FLAGS.num_inception_images,FLAGS.image_dir, 
                             FLAGS.eval_real_images)

  generator = tf.keras.models.load_model(FLAGS.model_dir +
                                         'Phase_2/interpolated_generator')
  data = data_provider.get_div2k_data(hparams, mode='valid')
  eval_lib.evaluate(hparams, generator, data)
  
if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)