/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {scalar, zeros} from '@tensorflow/tfjs-core';

import {LayerVariable} from '../variables';

import * as types_utils from './types_utils';

describe('isArrayOfShapes', () => {
  it('returns false for a single non-empty shape', () => {
    expect(types_utils.isArrayOfShapes([1, 2, 3])).toEqual(false);
  });
  it('returns false for a single empty shape', () => {
    expect(types_utils.isArrayOfShapes([])).toEqual(false);
  });
  it('returns true for an array of shapes', () => {
    expect(types_utils.isArrayOfShapes([[1], [2, 3]])).toEqual(true);
  });
  it('returns true for an array of shapes that includes empty shapes', () => {
    expect(types_utils.isArrayOfShapes([[], [2, 3]])).toEqual(true);
    expect(types_utils.isArrayOfShapes([[]])).toEqual(true);
    expect(types_utils.isArrayOfShapes([[], []])).toEqual(true);
  });
});

describe('normalizeShapeList', () => {
  it('returns an empty list if an empty list is passed in.', () => {
    expect(types_utils.normalizeShapeList([])).toEqual([]);
  });

  it('returns a list of shapes if a single shape is passed in.', () => {
    expect(types_utils.normalizeShapeList([1])).toEqual([[1]]);
  });

  it('returns a list of shapes if an empty shape is passed in.', () => {
    expect(types_utils.normalizeShapeList([[]])).toEqual([[]]);
  });

  it('returns a list of shapes if a list of shapes is passed in.', () => {
    expect(types_utils.normalizeShapeList([[1]])).toEqual([[1]]);
  });
});

describe('getExactlyOneShape', () => {
  it('single instance', () => {
    expect(types_utils.getExactlyOneShape([1, 2, 3])).toEqual([1, 2, 3]);
    expect(types_utils.getExactlyOneShape([null, 8])).toEqual([null, 8]);
    expect(types_utils.getExactlyOneShape([])).toEqual([]);
  });
  it('Array of length 1', () => {
    expect(types_utils.getExactlyOneShape([[1, 2]])).toEqual([1, 2]);
    expect(types_utils.getExactlyOneShape([[]])).toEqual([]);
  });
  it('Array of length 2: ValueError', () => {
    expect(() => types_utils.getExactlyOneShape([
      [1], [2]
    ])).toThrowError(/Expected exactly 1 Shape; got 2/);
  });
});

describe('countParamsInWeights', () => {
  it('Zero weights', () => {
    expect(types_utils.countParamsInWeights([])).toEqual(0);
  });

  it('One float32 weight', () => {
    const weight1 = new LayerVariable(zeros([2, 3]));
    expect(types_utils.countParamsInWeights([weight1])).toEqual(6);
  });

  it('One float32 scalar weight', () => {
    const weight1 = new LayerVariable(scalar(42));
    expect(types_utils.countParamsInWeights([weight1])).toEqual(1);
  });

  it('One int32 weight', () => {
    const weight1 = new LayerVariable(zeros([1, 3, 4], 'int32'), 'int32');
    expect(types_utils.countParamsInWeights([weight1])).toEqual(12);
  });

  it('Two weights, mixed types and shapes', () => {
    const weight1 = new LayerVariable(scalar(42));
    const weight2 = new LayerVariable(zeros([2, 3]));
    const weight3 = new LayerVariable(zeros([1, 3, 4], 'int32'), 'int32');
    expect(types_utils.countParamsInWeights([
      weight1, weight2, weight3
    ])).toEqual(19);
  });
});
