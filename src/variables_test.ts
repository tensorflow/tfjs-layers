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
 * Unit tests for LayerVariables.
 */

// tslint:disable:max-line-length
import {scalar, tensor1d, zeros} from '@tensorflow/tfjs-core';

import {nameScope} from './backend/tfjs_backend';
import * as tfl from './index';
import {DType} from './types';
import {describeMathCPU} from './utils/test_utils';
import {LayerVariable} from './variables';

// tslint:enable:max-line-length


/**
 * Unit tests for Variable.
 */
describeMathCPU('Variable', () => {
  it('Variable constructor: no explicit name', () => {
    const v1 = new LayerVariable(zeros([2]));
    expect(v1.name.indexOf('Variable')).toEqual(0);
    expect(v1.dtype).toEqual(DType.float32);
    expect(v1.shape).toEqual([2]);
    expect(v1.trainable).toEqual(true);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0, 0]));

    const v2 = new LayerVariable(zeros([2, 2]));
    expect(v2.name.indexOf('Variable')).toEqual(0);
    expect(v2.dtype).toEqual(DType.float32);
    expect(v2.shape).toEqual([2, 2]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));

    expect(v2.name === v1.name).toBe(false);
  });

  it('Variable constructor: explicit name', () => {
    const v1 = new LayerVariable(zeros([]), undefined, 'foo');
    expect(v1.name.indexOf('foo')).toEqual(0);
    expect(v1.dtype).toEqual(DType.float32);
    expect(v1.shape).toEqual([]);
    expect(v1.trainable).toEqual(true);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0]));

    const v2 = new LayerVariable(zeros([2, 2, 1]));
    expect(v1.name.indexOf('foo')).toEqual(0);
    expect(v2.dtype).toEqual(DType.float32);
    expect(v2.shape).toEqual([2, 2, 1]);
    expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    expect(v2.name.length).toBeGreaterThan(0);

    expect(v2.name === v1.name).toBe(false);
  });

  it('Variable constructor: explicit name with name scope', () => {
    let v1: LayerVariable;
    nameScope('barScope', () => {
      nameScope('bazScope', () => {
        v1 = new LayerVariable(scalar(0), undefined, 'foo');
      });
    });
    expect(v1.name.indexOf('barScope/bazScope/foo')).toEqual(0);
    expect(v1.dtype).toEqual(DType.float32);
    expect(v1.shape).toEqual([]);
    expect(v1.trainable).toEqual(true);
    expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
  });

  it('Variable trainable property', () => {
    const v1 = new LayerVariable(zeros([]), null, 'foo', false);
    expect(v1.trainable).toEqual(false);
  });

  it('Variable works if name is null or undefined', () => {
    expect((new LayerVariable(zeros([]), null)).name.indexOf('Variable'))
        .toEqual(0);
    expect((new LayerVariable(zeros([]), undefined)).name.indexOf('Variable'))
        .toEqual(0);
  });

  it('int32 dtype', () => {
    expect(new LayerVariable(zeros([]), DType.int32).dtype)
        .toEqual(DType.int32);
  });

  it('bool dtype', () => {
    expect(new LayerVariable(zeros([]), DType.bool).dtype).toEqual(DType.bool);
  });

  it('Read value', () => {
    const v1 = new LayerVariable(scalar(10), null, 'foo');
    expect(v1.read().dataSync()).toEqual(new Float32Array([10]));
  });

  it('Update value: Compatible shape', () => {
    const v = new LayerVariable(tensor1d([10, -10]), null, 'bar');
    expect(v.name.indexOf('bar')).toEqual(0);
    expect(v.shape).toEqual([2]);
    expect(v.read().dataSync()).toEqual(new Float32Array([10, -10]));

    v.write(tensor1d([10, 50]));
    expect(v.name.indexOf('bar')).toEqual(0);
    expect(v.shape).toEqual([2]);
    expect(v.read().dataSync()).toEqual(new Float32Array([10, 50]));
  });

  it('Update value: w/ constraint', () => {
    const v = new LayerVariable(
        tensor1d([10, -10]), null, 'bar', true, tfl.constraints.nonNeg());

    v.write(tensor1d([-10, 10]));
    expect(v.read().dataSync()).toEqual(new Float32Array([0, 10]));
  });


  it('Update value: Incompatible shape', () => {
    const v = new LayerVariable(zeros([2, 2]), null, 'qux');
    expect(() => {
      v.write(zeros([4]));
    }).toThrowError();
  });

  it('Generates unique ID', () => {
    const v1 = new LayerVariable(scalar(1), null, 'foo');
    const v2 = new LayerVariable(scalar(1), null, 'foo');
    expect(v1.id).not.toEqual(v2.id);
  });

  it('Generates unique IDs for Tensors and Variables', () => {
    const v1 = scalar(1);
    const v2 = new LayerVariable(scalar(1), null, 'foo');
    expect(v1.id).not.toEqual(v2.id);
  });
});
