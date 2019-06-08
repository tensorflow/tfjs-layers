/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {Tensor, tensor} from '@tensorflow/tfjs-core';
import {LossOrMetricFn} from '../types';
import * as util from './metric_utils';
import * as tfl from '../index';

describe('getLossOrMetricFnName', () => {
  it('string short cut name', async () => {
    const fnName = util.getLossOrMetricFnName('meanSquaredError');
    expect(fnName).toEqual('meanSquaredError');
  });

  it('function included in losses map', async () => {
    const fnName = util.getLossOrMetricFnName(tfl.metrics.meanSquaredError);
    expect(fnName).toEqual('meanSquaredError');
  });

  it('function included in metrics map', async () => {
    const fnName = util.getLossOrMetricFnName(tfl.metrics.categoricalAccuracy);
    expect(fnName).toEqual('categoricalAccuracy');
  });

  it('function not included in losses map or metrics map',
      async () => {
    const fakeMetric: LossOrMetricFn =
        (yTrue: Tensor, yPred: Tensor) => tensor([1]) as Tensor;
    const fnName = util.getLossOrMetricFnName(fakeMetric);
    expect(fnName).toEqual('fakeMetric');
  });

  it('throws null', async () => {
    expect(() => util.getLossOrMetricFnName(null)).toThrowError();
  });
});