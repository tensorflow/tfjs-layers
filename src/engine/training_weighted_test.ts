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

describeMathCPUAndGPU('LayersModel.fit() with classWeight', () => {
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
      classWeight: {0: 1, 1: 10, 2: 1}
    });
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(4.3944);
    expect(history.history.loss[1]).toBeCloseTo(5.3727);
  });

  // Reference Python code.
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     units=1,
  //     input_shape=[2],
  //     kernel_initializer='zeros',
  //     activation='sigmoid'))
  // model.compile(loss='binary_crossentropy',
  //               optimizer=tf.keras.optimizers.SGD(1.0))
  // model.summary()
  //
  // xs = np.array([[0, 1], [0, 2], [1, 10], [1, 20]],
  //               dtype=np.float32)
  // ys = np.array([[0], [0], [1], [1]], dtype=np.float32)
  //
  // # model.fit(xs, ys, epochs=1)
  // model.fit(xs,
  //           ys,
  //           epochs=3,
  //           class_weight=[{
  //               0: 0.1,
  //               1: 0.9
  //           }])
  // print(model.get_weights()[0])
  // ```
  it('One output, binary classes, sparse encoding', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 1,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'sigmoid'
    }));
    model.compile({
      loss: 'binaryCrossentropy',
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20]]);
    const ys = tensor2d([[0], [0], [1], [1]]);
    const history = await model.fit(xs, ys, {
      epochs: 2,
      classWeight: [{0: 0.1, 1: 0.9}]
    });
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(0.3466);
    expect(history.history.loss[1]).toBeCloseTo(0.2611);
  });

  // Python Reference Code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // inp = tf.keras.Input(shape=[2])
  // y1 = tf.keras.layers.Dense(
  //     units=3,
  //     input_shape=[2],
  //     kernel_initializer='zeros',
  //     activation='softmax')(inp)
  // y2 = tf.keras.layers.Dense(
  //     units=1,
  //     input_shape=[2],
  //     kernel_initializer='zeros',
  //     activation='sigmoid')(inp)
  // model = tf.keras.Model(inp, [y1, y2])
  // model.compile(
  //     loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
  //     optimizer=tf.keras.optimizers.SGD(1.0))
  // model.summary()
  //
  // xs = np.array(
  //     [[0, 1], [0, 2], [1, 10], [1, 20], [2, 10], [2, 20]],
  //     dtype=np.float32)
  // y1s = np.array(
  //     [[0], [0], [1], [1], [2], [2]], dtype=np.float32)
  // y2s = np.array(
  //     [[0], [0], [1], [1], [1], [1]], dtype=np.float32)
  //
  // # model.fit(xs, ys, epochs=1)
  // model.fit(xs,
  //           [y1s, y2s],
  //           epochs=3,
  //           class_weight=[{
  //               0: 0.1,
  //               1: 0.2,
  //               2: 0.7
  //           }, {
  //               0: 0.1,
  //               1: 0.9
  //           }])
  // ```
  it('Two outputs, classWeight as array' , async () => {
    const inp = tfl.input({shape: [2]});
    const y1 = tfl.layers.dense({
      units: 3,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'softmax'
    }).apply(inp) as tfl.SymbolicTensor;
    const y2 = tfl.layers.dense({
      units: 1,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'sigmoid'
    }).apply(inp) as tfl.SymbolicTensor;
    const model = tfl.model({
      inputs: inp,
      outputs: [y1, y2]
    });
    model.compile({
      loss: ['sparseCategoricalCrossentropy', 'binaryCrossentropy'],
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, 10], [2, 20]]);
    const y1s = tensor2d([[0], [0], [1], [1], [2], [2]]);
    const y2s = tensor2d([[0], [0], [1], [1], [1], [1]]);

    const history = await model.fit(xs, [y1s, y2s], {
      epochs: 3,
      classWeight: [{0: 0.1, 1: 0.2, 2: 0.7}, {0: 0.1, 1: 0.9}]
    });
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.loss[0]).toBeCloseTo(0.8052);
    expect(history.history.loss[1]).toBeCloseTo(1.4887);
    expect(history.history.loss[2]).toBeCloseTo(1.4782);
    const lossKey0 = `${model.outputNames[0]}_loss`;
    expect(history.history[lossKey0].length).toEqual(3);
    expect(history.history[lossKey0][0]).toBeCloseTo(0.3662);
    expect(history.history[lossKey0][1]).toBeCloseTo(1.2553);
    expect(history.history[lossKey0][2]).toBeCloseTo(1.2485);
    const lossKey1 = `${model.outputNames[1]}_loss`;
    expect(history.history[lossKey1].length).toEqual(3);
    expect(history.history[lossKey1][0]).toBeCloseTo(0.4390);
    expect(history.history[lossKey1][1]).toBeCloseTo(0.2333);
    expect(history.history[lossKey1][2]).toBeCloseTo(0.2297);
  });

  // TODO(cais): classWeight as dict missing key.
  // TODO(cais): classWeight with a null element.
  // TODO(cais): fitDataset with classWeight.
  // TODO(cais): Metrics.
  // TODO(cais): Validation data.
});
