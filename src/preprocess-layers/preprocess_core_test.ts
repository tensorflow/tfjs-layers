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
 * Unit tests for core.ts.
 */

// tslint:disable:max-line-length
import {eye, Tensor, tensor1d, zeros} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {describeMathGPU, expectTensorsClose} from '../utils/test_utils';

// tslint:enable:max-line-length

describe('OneHot Layer: Symbolic', () => {
  const units = 30;
  const symbolicInput = new tfl.SymbolicTensor('int32', [10], null, [], null);
  const testTitle = `units=${units}; ` +
      `input shape=${JSON.stringify(symbolicInput.shape)}`;
  it(testTitle, () => {
    const oneHotLayer = tfl.layers.oneHot({units});
    const output = oneHotLayer.apply(symbolicInput) as tfl.SymbolicTensor;
    expect(output.dtype).toEqual('float32');
    expect(output.shape).toEqual([10, 30]);
    expect(output.sourceLayer).toEqual(oneHotLayer);
    expect(output.inputs).toEqual([symbolicInput]);
  });
});


// TODO(bileschi): Replace with describeMathCPUandGPU when #443 is resolved.
describeMathGPU('OneHot Layer: Tensor', () => {
  it('handles in-range integer inputs', () => {
    const units = 5;
    const x = tensor1d([0, 1, 2, 3, 4], 'int32');
    const oneHotLayer = tfl.layers.oneHot({units});
    const y = oneHotLayer.apply(x) as Tensor;
    expect([5, 5]).toEqual(y.shape);
    const expectedOutput = eye(units);
    expectTensorsClose(y, expectedOutput);
  });
  it('handles out-of-range integer inputs', () => {
    const units = 10;
    // TODO(bileschi): Add a test here for NaN support after #442 is resolved.
    const sampleInput = [-1, 29999, units, 30000, 30001];
    const x = tensor1d(sampleInput, 'int32');
    const oneHotLayer = tfl.layers.oneHot({units});
    const y = oneHotLayer.apply(x) as Tensor;
    expect([sampleInput.length, units]).toEqual(y.shape);
    const expectedOutput = zeros([sampleInput.length, units]);
    expectTensorsClose(y, expectedOutput);
  });
});
