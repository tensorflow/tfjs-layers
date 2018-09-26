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
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    return model;
  }

  // TODO(cais): Test error message for uncompiled models.
  // TODO(cais): Test for memory leaks.

  it('No validation', async () => {
    const model = createDenseModel();

    const batchSize = 8;
    const numBatches = 3;
    const epochs = 2;
    const presetXTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const presetYTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches,
      presetXTensorsFunc,
      presetYTensorsFunc
    });
    const history =
        await model.fitDataset(dataset, {stepsPerEpoch: numBatches, epochs});
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.loss[0]).toBeCloseTo(0.923649);
    expect(history.history.loss[1]).toBeCloseTo(0.722993);
    expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
    expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
  });
});
