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
import {eye, Tensor, tensor1d, util, zeros} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from '../utils/test_utils';

import {flattenStringArray, StringArray, StringTensor} from './preprocess_core';

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


// TODO(bileschi): Expand the version in core to accept StringTensor and
// string[];
export function stringArraysEqual(n1: string[], n2: string[]) {
  if (n1.length !== n2.length) {
    return false;
  }
  for (let i = 0; i < n1.length; i++) {
    if (n1[i] !== n2[i]) {
      return false;
    }
  }
  return true;
}

// TODO(bileschi): Expand eversion in core to accept StringTensor and string[];
export function expectStringArraysMatch(
    actual: StringTensor|StringArray, expected: StringTensor|StringArray) {
  if (!(actual instanceof StringTensor) &&
      !(expected instanceof StringTensor)) {
    const aType = actual.constructor.name;
    const bType = expected.constructor.name;

    if (aType !== bType) {
      throw new Error(
          `Arrays are of different type actual: ${aType} ` +
          `vs expected: ${bType}`);
    }
  } else if (
      actual instanceof StringTensor && expected instanceof StringTensor) {
    if (!util.arraysEqual(actual.shape, expected.shape)) {
      throw new Error(
          `Arrays are of different shape actual: ${actual.shape} ` +
          `vs expected: ${expected.shape}.`);
    }
  }

  let actualValues: string[];
  let expectedValues: string[];
  if (actual instanceof StringTensor) {
    actualValues = actual.stringValues;
  } else {
    actualValues = flattenStringArray(actual);
  }
  if (expected instanceof StringTensor) {
    expectedValues = expected.stringValues;
  } else {
    expectedValues = flattenStringArray(expected);
  }

  if (actualValues.length !== expectedValues.length) {
    throw new Error(
        `Arrays have different lengths actual: ${actualValues.length} vs ` +
        `expected: ${expectedValues.length}.\n` +
        `Actual:   ${actualValues}.\n` +
        `Expected: ${expectedValues}.`);
  }
  for (let i = 0; i < expectedValues.length; ++i) {
    const a = actualValues[i];
    const e = expectedValues[i];

    if (a !== e) {
      throw new Error(
          `Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
          `Actual:   ${actualValues}.\n` +
          `Expected: ${expectedValues}.`);
    }
  }
}


describeMathCPUAndGPU('String Tensor', () => {
  it('can create a string tensor', () => {
    const st: StringTensor = tfl.preprocessing.stringTensor2D(
        [['hello', 'world'], ['mellow', 'morld']], [2, 2]);
    expect(st.rank).toBe(2);
    expect(st.size).toBe(4);
    expectStringArraysMatch(st, ['hello', 'world', 'mellow', 'morld']);
    // Out of bounds indexing.
    expect(st.get(3, 3)).toBeUndefined();
  });
});
