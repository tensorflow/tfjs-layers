# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import numpy as np
import tensorflow as tf


class TfKeras2TfjsTest(tf.test.TestCase):

  def testFoo(self):
    print(tf.__version__)
    save_target_dir = tempfile.mkdtemp()

    # TODO(cais): Rename test.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=[10], activation='sigmoid'))
    model.summary()
    saved_to_path = tf.contrib.saved_model.save_keras_model(
        model, save_target_dir)

    model2 = tf.contrib.saved_model.load_keras_model(saved_to_path)
    model2.summary()
    model2.save('/tmp/qux_model.h5')

    # TODO(cais): Implement in tensorflowjs --input_format tf_keras_saved_model
    #   --output_format tfjs <in_path> <out_path>
    print('saved_to_path = %s' % saved_to_path)  # DEBUG

if __name__ == '__main__':
  tf.test.main()
