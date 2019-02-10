/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
import {Tensor} from '@tensorflow/tfjs-core';

import {describeMathCPUAndGPU} from '../utils/test_utils';
import {FakeNumericDataset} from './dataset_fakes';
import {TensorMap} from './dataset_stub';

describeMathCPUAndGPU('FakeNumericDataset', () => {
  it('1D features, 1D targets', async () => {
    const dataset = new FakeNumericDataset(
        {xShape: [3], yShape: [1], batchSize: 8, numBatches: 5});
    for (let k = 0; k < 2; ++k) {
      // Run twice to make sure that calling iteartor() multiple times works.
      const iterator = await dataset.iterator();
      for (let i = 0; i < 5; ++i) {
        const result = await iterator.next();
        expect(result.value.length).toEqual(2);
        expect((result.value[0] as Tensor).shape).toEqual([8, 3]);
        expect((result.value[1] as Tensor).shape).toEqual([8, 1]);
        expect(result.done).toEqual(false);
      }
      for (let i = 0; i < 3; ++i) {
        const result = await iterator.next();
        expect(result.value).toBeNull();
        expect(result.done).toEqual(true);
      }
    }
  });

  it('2D features, 1D targets', async () => {
    const dataset = new FakeNumericDataset(
        {xShape: [3, 4], yShape: [2], batchSize: 8, numBatches: 5});
    for (let k = 0; k < 2; ++k) {
      // Run twice to make sure that calling iteartor() multiple times works.
      const iterator = await dataset.iterator();
      for (let i = 0; i < 5; ++i) {
        const result = await iterator.next();
        expect(result.value.length).toEqual(2);
        expect((result.value[0] as Tensor).shape).toEqual([8, 3, 4]);
        expect((result.value[1] as Tensor).shape).toEqual([8, 2]);
        expect(result.done).toEqual(false);
      }
      for (let i = 0; i < 3; ++i) {
        const result = await iterator.next();
        expect(result.value).toBeNull();
        expect(result.done).toEqual(true);
      }
    }
  });

  it('Multiple 2D features, 1D targets', async () => {
    const dataset = new FakeNumericDataset({
      xShape: {'input1': [3, 4], 'input2': [2, 3]},
      yShape: [2],
      batchSize: 8,
      numBatches: 5
    });
    for (let k = 0; k < 2; ++k) {
      // Run twice to make sure that calling iteartor() multiple times works.
      const iterator = await dataset.iterator();
      for (let i = 0; i < 5; ++i) {
        const result = await iterator.next();
        expect(result.value.length).toEqual(2);
        const xs = result.value[0] as TensorMap;
        expect(xs['input1'].shape).toEqual([8, 3, 4]);
        expect(xs['input2'].shape).toEqual([8, 2, 3]);
        expect((result.value[1] as Tensor).shape).toEqual([8, 2]);
        expect(result.done).toEqual(false);
      }
      for (let i = 0; i < 3; ++i) {
        const result = await iterator.next();
        expect(result.value).toBeNull();
        expect(result.done).toEqual(true);
      }
    }
  });

  it('Multiple 1D features, 1D targets, with tensors function', async () => {

    const batchSize = 8;

    // Training data.
    // Feature with different shapes
    const xTensorsFunction = () => {
      const output: {[name: string]: tfc.Tensor[]} = {};
      output['input1'] = [
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
      ];
      output['input2'] = [
        tfc.ones([batchSize, 2]), tfc.ones([batchSize, 2]),
        tfc.ones([batchSize, 2]), tfc.ones([batchSize, 2]),
        tfc.ones([batchSize, 2]), tfc.ones([batchSize, 2])
      ];
      return output;
    };
    const yTensorsFunction = () => [
      tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
      tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
      tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
    ];

    const xShapes: {[name:string]: number[]} = {};
    xShapes['input1'] = [1];
    xShapes['input2'] = [2];
    const dataset = new FakeNumericDataset({
      xShape: xShapes,
      yShape: [1],
      batchSize: 8,
      numBatches: 6,
      xTensorsFunc: xTensorsFunction,
      yTensorsFunc: yTensorsFunction
    });

    for (let k = 0; k < 2; ++k) {
      // Run twice to make sure that calling iteartor() multiple times works.

      const numTensors0 = tfc.memory().numTensors;
      const iterator = await dataset.iterator();
      for (let i = 0; i < 6; ++i) {
        const result = await iterator.next();
        expect(result.value.length).toEqual(2);
        const xs = result.value[0] as TensorMap;
        expect(xs['input1'].shape).toEqual([8, 1]);
        const xsInput1Data = (xs['input1'] as Tensor).dataSync();
        expect(xsInput1Data[0]).toEqual(1);
        expect(xs['input2'].shape).toEqual([8, 2]);
        const xsInput2Data = (xs['input2'] as Tensor).dataSync();
        expect(xsInput2Data[0]).toEqual(1);
        expect(xsInput2Data[1]).toEqual(1);
        const ys = result.value[1] as Tensor;
        expect(ys.shape).toEqual([8, 1]);
        const ysData = ys.dataSync();
        expect(ysData[0]).toEqual(1);
        expect(result.done).toEqual(false);
        tfc.dispose(result.value);
      }
      for (let i = 0; i < 3; ++i) {
        const result = await iterator.next();
        expect(result.value).toBeNull();
        expect(result.done).toEqual(true);
      }

      // Make sure no memory leak
      const numTensors1 = tfc.memory().numTensors;
      expect(numTensors1).toEqual(numTensors0);
    }

  });

  it('1D features, multiple 1D targets, with tensors function', async () => {

    const batchSize = 8;

    // Training data.
    const xTensorsFunction = () => [
      tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
      tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
      tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
    ];
    // Target with different shapes
    const yTensorsFunction = () => {
      const output: {[name: string]: tfc.Tensor[]} = {};
      output['output1'] = [
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
      ];
      output['output2'] = [
        tfc.ones([batchSize, 2]), tfc.ones([batchSize, 2]),
        tfc.ones([batchSize, 2]), tfc.ones([batchSize, 2]),
        tfc.ones([batchSize, 2]), tfc.ones([batchSize, 2])
      ];
      return output;
    };

    const yShapes: {[name:string]: number[]} = {};
    yShapes['output1'] = [1];
    yShapes['output2'] = [2];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: yShapes,
      batchSize: 8,
      numBatches: 6,
      xTensorsFunc: xTensorsFunction,
      yTensorsFunc: yTensorsFunction
    });

    for (let k = 0; k < 2; ++k) {
      // Run twice to make sure that calling iteartor() multiple times works.

      const numTensors0 = tfc.memory().numTensors;
      const iterator = await dataset.iterator();
      for (let i = 0; i < 6; ++i) {
        const result = await iterator.next();
        expect(result.value.length).toEqual(2);
        const xs = result.value[0] as Tensor;
        expect(xs.shape).toEqual([8, 1]);
        const xsData = xs.dataSync();
        expect(xsData[0]).toEqual(1);
        const ys = result.value[1] as TensorMap;
        expect(ys['output1'].shape).toEqual([8, 1]);
        const ysOutput1Data = (ys['output1'] as Tensor).dataSync();
        expect(ysOutput1Data[0]).toEqual(1);
        expect(ys['output2'].shape).toEqual([8, 2]);
        const ysOutput2Data = (ys['output2'] as Tensor).dataSync();
        expect(ysOutput2Data[0]).toEqual(1);
        expect(ysOutput2Data[1]).toEqual(1);
        tfc.dispose(result.value);
      }
      for (let i = 0; i < 3; ++i) {
        const result = await iterator.next();
        expect(result.value).toBeNull();
        expect(result.done).toEqual(true);
      }

      // Make sure no memory leak
      const numTensors1 = tfc.memory().numTensors;
      expect(numTensors1).toEqual(numTensors0);
    }

  });

  it('Invalid batchSize leads to Error', () => {
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: -8, numBatches: 5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 8.5, numBatches: 5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 0, numBatches: 5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            // tslint:disable-next-line:no-any
            {xShape: [3], yShape: [1], batchSize: 'foo' as any, numBatches: 5}))
        .toThrow();
  });

  it('Invalid numBatches leads to Error', () => {
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 8, numBatches: -5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 8, numBatches: 5.5}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            {xShape: [3], yShape: [1], batchSize: 8, numBatches: 0}))
        .toThrow();
    expect(
        () => new FakeNumericDataset(
            // tslint:disable-next-line:no-any
            {xShape: [3], yShape: [1], batchSize: 8, numBatches: 'foo' as any}))
        .toThrow();
  });
});
