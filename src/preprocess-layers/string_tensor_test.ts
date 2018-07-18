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
import {util} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {describeMathCPUAndGPU} from '../utils/test_utils';

import {flattenStringArray, StringArray, StringTensor} from './string_tensor';

// tslint:enable:max-line-length

////////////////////////////////////////////////////////
//////////////TEST UTIL STUFF///////////////////////////
////////////////////////////////////////////////////////

// TODO(bileschi): Expand the 'expectArraysMatch  in core to accept
// StringTensor and string[] and use that instead of this.  Delete this.
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

////////////////////////////////////////////////////////
////////////// ACTUAL TESTS ////////////////////////////
////////////////////////////////////////////////////////

describeMathCPUAndGPU('String Tensor Creation', () => {
  it('1D', () => {
    const st: StringTensor =
        tfl.preprocessing.stringTensor1d(['hello', 'world']);
    expect(st.rank).toBe(1);
    expect(st.size).toBe(2);
    expectStringArraysMatch(st, ['hello', 'world']);
    expect(st.get(1)).toBe('world');
    // Out of bounds indexing.
    expect(st.get(78)).toBeUndefined();
  });

  it('2D', () => {
    const st: StringTensor = tfl.preprocessing.stringTensor2d(
        [['hello', 'world'], ['こんにちは', '世界']], [2, 2]);
    expect(st.rank).toBe(2);
    expect(st.size).toBe(4);
    expectStringArraysMatch(st, ['hello', 'world', 'こんにちは', '世界']);
    expect(st.get(1, 1)).toBe('世界');
    // Out of bounds indexing.
    expect(st.get(3, 3)).toBeUndefined();
  });

  it('3D', () => {
    const st: StringTensor = tfl.preprocessing.stringTensor3d(
        [[['hello'], ['world']], [['こんにちは'], ['世界']]], [2, 2, 1]);
    expect(st.rank).toBe(3);
    expect(st.size).toBe(4);
    expectStringArraysMatch(st, ['hello', 'world', 'こんにちは', '世界']);
    expect(st.get(0, 0, 0)).toBe('hello');
    // Out of bounds indexing.
    expect(st.get(3, 3, 9)).toBeUndefined();
  });

  it('4D', () => {
    const st: StringTensor = tfl.preprocessing.stringTensor4d(
        [[[['hello'], ['world']], [['こんにちは'], ['世界']]]], [1, 2, 2, 1]);
    expect(st.rank).toBe(4);
    expect(st.size).toBe(4);
    expectStringArraysMatch(st, ['hello', 'world', 'こんにちは', '世界']);
    expect(st.get(0, 0, 1, 0)).toBe('world');
    // Out of bounds indexing.
    expect(st.get(11, 9, 19, 78)).toBeUndefined();
  });

  it('5D', () => {
    const st: StringTensor = tfl.preprocessing.stringTensor5d(
        [[[[['hello'], ['world']], [['こんにちは'], ['世界']]]]],
        [1, 1, 2, 2, 1]);
    expect(st.rank).toBe(5);
    expect(st.size).toBe(4);
    expectStringArraysMatch(st, ['hello', 'world', 'こんにちは', '世界']);
    expect(st.get(0, 0, 1, 0, 0)).toBe('こんにちは');
    // Out of bounds indexing.
    expect(st.get(0, 0, 3, 3, 9)).toBeUndefined();
  });

  it('6D', () => {
    const st: StringTensor = tfl.preprocessing.stringTensor6d(
        [[[[[['hello'], ['world']], [['こんにちは'], ['世界']]]]]],
        [1, 1, 1, 2, 2, 1]);
    expect(st.rank).toBe(6);
    expect(st.size).toBe(4);
    expectStringArraysMatch(st, ['hello', 'world', 'こんにちは', '世界']);
    expect(st.get(0, 0, 0, 0, 0, 0)).toBe('hello');
    // Out of bounds indexing.
    expect(st.get(0, 0, 0, 3, 3, 9)).toBeUndefined();
  });
});

describeMathCPUAndGPU('String Tensor Reshape', () => {
  it('2D', () => {
    const stOrig: StringTensor = tfl.preprocessing.stringTensor2d(
        [['hello', 'world'], ['こんにちは', '世界']], [2, 2]);
    const st = stOrig.reshape([1, 1, 4, 1, 1]);
    expect(st.rank).toBe(5);
    expect(st.size).toBe(4);
    expectStringArraysMatch(st, ['hello', 'world', 'こんにちは', '世界']);
    expect(st.get(0, 0, 3, 0, 0)).toBe('世界');
  });
});


describeMathCPUAndGPU('String Tensor set value', () => {
  it('2D', () => {
    const st: StringTensor = tfl.preprocessing.stringTensor2d(
        [['hello', 'world'], ['こんにちは', 'TODO...']], [2, 2]);
    st.set('世界', 1, 1);
    expect(st.rank).toBe(2);
    expect(st.size).toBe(4);
    expectStringArraysMatch(st, ['hello', 'world', 'こんにちは', '世界']);
    expect(st.get(1, 1)).toBe('世界');
    // Out of bounds indexing.
    expect(st.get(3, 3)).toBeUndefined();
  });
});
