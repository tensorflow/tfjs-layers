/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/** Unit tests for training with {sample, class} weights. */

import {tensor2d, train} from '@tensorflow/tfjs-core';

import * as tfl from '../index';

import {describeMathCPUAndGPU} from '../utils/test_utils';

describeMathCPUAndGPU('fit() with classWeight', () => {
  // Reference Python code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     units=3,
  //     input_shape=[2],
  //     kernel_initializer='zeros',
  //     activation='softmax'))
  // model.compile(loss='categorical_crossentropy',
  //               optimizer=tf.keras.optimizers.SGD(1.0))
  // model.summary()
  //
  // xs = np.array([[0, 1], [0, 2], [1, 10], [1, 20], [2, -10], [2, -20]],
  //               dtype=np.float32)
  // ys = np.array([[1, 0, 0],
  //                [1, 0, 0],
  //                [0, 1, 0],
  //                [0, 1, 0],
  //                [0, 0, 1],
  //                [0, 0, 1]], dtype=np.float32)
  //
  // model.fit(xs,
  //           ys,
  //           epochs=2,
  //           class_weight=[{
  //             0: 1,
  //             1: 10,
  //             2: 1
  //           }])
  // print(model.get_weights()[0])
  // ```
  it('One output, multi-class, one-hot encoding', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'softmax'
    }));
    model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, -10], [2, -20]]);
    const ys = tensor2d([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1]]);
    const history = await model.fit(xs, ys, {
      epochs: 2,
      classWeight: [{0: 1, 1: 10, 2: 1}]
    });
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(4.3944);
    expect(history.history.loss[1]).toBeCloseTo(5.3727);
  });

  it('One output, multi-class, sparse encoding', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'softmax'
    }));
    model.compile({
      loss: 'sparseCategoricalCrossentropy',
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, -10], [2, -20]]);
    const ys = tensor2d([[0], [0], [1], [1], [2], [2]]);
    const history = await model.fit(xs, ys, {
      epochs: 2,
      classWeight: [{0: 1, 1: 10, 2: 1}]
    });
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(4.3944);
    expect(history.history.loss[1]).toBeCloseTo(5.3727);
  });
});
