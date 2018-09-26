/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit tests for training.ts, focusing on the tf.Model.fitDataset() method.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import * as tfl from '../index';
import {describeMathCPUAndGPU} from '../utils/test_utils';

import {FakeNumericDataset} from './dataset_stub';

describeMathCPUAndGPU('Model.fitDataset', () => {
  function createDenseModel(): tfl.Model {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 1,
      inputShape: [1],
      kernelInitializer: 'zeros',
      biasInitializer: 'zeros'
    }));
    return model;
  }

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros',
  //     bias_initializer='zeros'))
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // history = model.fit(dataset, steps_per_epoch=num_batches, epochs=epochs)
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // ```
  it('1 input, 1 output, no metric, no validation', async () => {
    const model = createDenseModel();
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const batchSize = 8;
    const epochs = 2;
    const stepsPerEpoch = 3;
    const presetXTensorsFunc = () =>
        [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
    const presetYTensorsFunc = () =>
        [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: stepsPerEpoch * epochs,
      presetXTensorsFunc,
      presetYTensorsFunc
    });

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    await model.fitDataset(dataset, {stepsPerEpoch, epochs: 1});
    model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

    const numTensors0 = tfc.memory().numTensors;
    const history = await model.fitDataset(dataset, {stepsPerEpoch, epochs});
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);
    expect(Object.keys(history.history)).toEqual(['loss']);
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.loss[0]).toBeCloseTo(0.923649);
    expect(history.history.loss[1]).toBeCloseTo(0.722993);
    expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
    expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
  });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros',
  //     bias_initializer='zeros'))
  // model.compile(loss='mean_squared_error',
  //               optimizer='sgd',
  //               metrics=['acc'])
  //
  // history = model.fit(dataset, steps_per_epoch=num_batches, epochs=epochs)
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // ```
  it('1 input, 1 output, 1 metric, no validation', async () => {
    const model = createDenseModel();
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

    const batchSize = 8;
    const epochs = 2;
    const stepsPerEpoch = 3;
    const presetXTensorsFunc = () =>
        [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
    const presetYTensorsFunc = () =>
        [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: stepsPerEpoch * epochs,
      presetXTensorsFunc,
      presetYTensorsFunc
    });

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    await model.fitDataset(dataset, {stepsPerEpoch, epochs: 1});
    model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

    const numTensors0 = tfc.memory().numTensors;
    const history = await model.fitDataset(dataset, {stepsPerEpoch, epochs});
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);
    expect(Object.keys(history.history)).toEqual(['loss', 'acc']);
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.loss[0]).toBeCloseTo(0.923649);
    expect(history.history.loss[1]).toBeCloseTo(0.722993);
    expect(history.history.acc.length).toEqual(2);
    expect(history.history.acc[0]).toBeCloseTo(0);
    expect(history.history.acc[1]).toBeCloseTo(0);
    expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
    expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
  });

  // TODO(cais): Test for dataset exhaustion.

  it('Calling fitDataset() without calling compile() errors', async () => {
    const model = createDenseModel();

    const batchSize = 8;
    const numBatches = 3;
    const epochs = 2;
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches,
    });

    let errorCaught: Error;
    try {
      await model.fitDataset(dataset, {stepsPerEpoch: numBatches, epochs});
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message)
        .toEqual('The model needs to be compiled before being used.');
  });
});
