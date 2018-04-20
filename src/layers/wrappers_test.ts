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
 * Unit tests for wrapper layers.
 */

// tslint:disable:max-line-length
import {Tensor, tensor2d, Tensor3D, tensor3d} from '@tensorflow/tfjs-core';

import {Layer} from '../engine/topology';
import * as tfl from '../index';
import {DType} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {Dense, Reshape} from './core';
import {SimpleRNN} from './recurrent';
import {BidirectionalMergeMode, TimeDistributed} from './wrappers';

// tslint:enable:max-line-length

describeMathCPU('TimeDistributed Layer: Symbolic', () => {
  it('3D input: Dense', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [10, 8, 2], null, [], null);
    const wrapper = tfl.layers.timeDistributed({layer: new Dense({units: 3})});
    const output = wrapper.apply(input) as tfl.SymbolicTensor;
    expect(wrapper.trainable).toEqual(true);
    expect(wrapper.getWeights().length).toEqual(2);  // kernel and bias.
    expect(output.dtype).toEqual(input.dtype);
    expect(output.shape).toEqual([10, 8, 3]);
  });
  it('4D input: Reshape', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [10, 8, 2, 3], null, [], null);
    const wrapper =
        tfl.layers.timeDistributed({layer: new Reshape({targetShape: [6]})});
    const output = wrapper.apply(input) as tfl.SymbolicTensor;
    expect(output.dtype).toEqual(input.dtype);
    expect(output.shape).toEqual([10, 8, 6]);
  });
  it('2D input leads to exception', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [10, 2], null, [], null);
    const wrapper = tfl.layers.timeDistributed({layer: new Dense({units: 3})});
    expect(() => wrapper.apply(input))
        .toThrowError(
            /TimeDistributed .*expects an input shape >= 3D, .* \[10,.*2\]/);
  });
  it('getConfig and fromConfig: round trip', () => {
    const wrapper = tfl.layers.timeDistributed({layer: new Dense({units: 3})});
    const config = wrapper.getConfig();
    const wrapperPrime = TimeDistributed.fromConfig(TimeDistributed, config);
    expect(wrapperPrime.getConfig()).toEqual(wrapper.getConfig());
  });
});

describeMathCPUAndGPU('TimeDistributed Layer: Tensor', () => {
  it('3D input: Dense', () => {
    const input = tensor3d(
        [
          [[1, 2], [3, 4], [5, 6], [7, 8]],
          [[-1, -2], [-3, -4], [-5, -6], [-7, -8]]
        ],
        [2, 4, 2]);
    // Given an all-ones Dense kernel and no bias, the output at each timestep
    // is expected to be [3, 7, 11, 15], give or take a minus sign.
    const wrapper = tfl.layers.timeDistributed({
      layer: new Dense({units: 1, kernelInitializer: 'ones', useBias: false})
    });
    const output = wrapper.apply(input) as Tensor;
    expectTensorsClose(
        output,
        tensor3d(
            [[[3], [7], [11], [15]], [[-3], [-7], [-11], [-15]]], [2, 4, 1]));
  });
});

describeMathCPU('Bidirectional Layer: Symbolic', () => {
  const mergeModes: BidirectionalMergeMode[] = [
    null,
    BidirectionalMergeMode.CONCAT,
    BidirectionalMergeMode.AVE,
    BidirectionalMergeMode.MUL,
    BidirectionalMergeMode.SUM,
  ];
  const returnStateValues: boolean[] = [false, true];

  for (const mergeMode of mergeModes) {
    for (const returnState of returnStateValues) {
      const testTitle = `3D input: returnSequence=false, ` +
          `mergeMode=${mergeMode}; returnState=${returnState}`;
      it(testTitle, () => {
        const input =
            new tfl.SymbolicTensor(DType.float32, [10, 8, 2], null, [], null);
        const bidi = tfl.layers.bidirectional({
          layer: new SimpleRNN(
              {units: 3, recurrentInitializer: 'glorotNormal', returnState}),
          mergeMode,
        });
        // TODO(cais): Remove recurrentInitializer once Orthogonal initializer
        // is available.
        let outputs = bidi.apply(input);
        expect(bidi.trainable).toEqual(true);
        // {kernel, recurrentKernel, bias} * {forward, backward}.
        expect(bidi.getWeights().length).toEqual(6);
        if (!returnState) {
          if (mergeMode === null) {
            outputs = outputs as tfl.SymbolicTensor[];
            expect(outputs.length).toEqual(2);
            expect(outputs[0].shape).toEqual([10, 3]);
            expect(outputs[1].shape).toEqual([10, 3]);
          } else if (mergeMode === BidirectionalMergeMode.CONCAT) {
            outputs = outputs as tfl.SymbolicTensor;
            expect(outputs.shape).toEqual([10, 6]);
          } else {
            outputs = outputs as tfl.SymbolicTensor;
            expect(outputs.shape).toEqual([10, 3]);
          }
        } else {
          if (mergeMode === null) {
            outputs = outputs as tfl.SymbolicTensor[];
            expect(outputs.length).toEqual(4);
            expect(outputs[0].shape).toEqual([10, 3]);
            expect(outputs[1].shape).toEqual([10, 3]);
            expect(outputs[2].shape).toEqual([10, 3]);
            expect(outputs[3].shape).toEqual([10, 3]);
          } else if (mergeMode === BidirectionalMergeMode.CONCAT) {
            outputs = outputs as tfl.SymbolicTensor[];
            expect(outputs.length).toEqual(3);
            expect(outputs[0].shape).toEqual([10, 6]);
            expect(outputs[1].shape).toEqual([10, 3]);
            expect(outputs[2].shape).toEqual([10, 3]);
          } else {
            outputs = outputs as tfl.SymbolicTensor[];
            expect(outputs.length).toEqual(3);
            expect(outputs[0].shape).toEqual([10, 3]);
            expect(outputs[1].shape).toEqual([10, 3]);
            expect(outputs[2].shape).toEqual([10, 3]);
          }
        }
      });
    }
  }
  it('returnSequence=true', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [10, 8, 2], null, [], null);
    const bidi = tfl.layers.bidirectional({
      layer: new SimpleRNN({
        units: 3,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true,
        returnState: true
      }),
      mergeMode: BidirectionalMergeMode.AVE
    });
    const outputs = bidi.apply(input) as tfl.SymbolicTensor[];
    expect(outputs.length).toEqual(3);
    expect(outputs[0].shape).toEqual([10, 8, 3]);
    expect(outputs[1].shape).toEqual([10, 3]);
    expect(outputs[2].shape).toEqual([10, 3]);
  });
});

describeMathCPUAndGPU('Bidirectional Layer: Tensor', () => {
  // The golden tensor values used in the tests below can be obtained with
  // PyKeras code such as the following:
  // ```python
  // import keras
  // import numpy as np
  //
  // bidi = keras.layers.Bidirectional(
  //     keras.layers.SimpleRNN(
  //         3,
  //         kernel_initializer='ones',
  //         recurrent_initializer='ones',
  //         return_state=True),
  //     merge_mode='ave')
  //
  // time_steps = 4
  // input_size = 2
  // inputs = keras.Input([time_steps, input_size])
  // outputs = bidi(inputs)
  // model = keras.Model(inputs, outputs)
  //
  // x = np.array(
  //     [[[0.05, 0.05],
  //       [-0.05, -0.05],
  //       [0.1, 0.1],
  //       [-0.1, -0.1]]])
  // print(model.predict(x))
  // ```

  // TODO(bileschi): This should be tfl.layers.Layer.
  let bidi: Layer;
  let x: Tensor3D;
  function createLayerAndData(
      mergeMode: BidirectionalMergeMode, returnState: boolean) {
    const units = 3;
    bidi = tfl.layers.bidirectional({
      layer: new SimpleRNN({
        units,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        useBias: false,
        returnState
      }),
      mergeMode,
    });
    const timeSteps = 4;
    const inputSize = 2;
    x = tensor3d(
        [[[0.05, 0.05], [-0.05, -0.05], [0.1, 0.1], [-0.1, -0.1]]],
        [1, timeSteps, inputSize]);
  }

  const mergeModes: BidirectionalMergeMode[] = [
    null,
    BidirectionalMergeMode.CONCAT,
    BidirectionalMergeMode.MUL,
  ];
  for (const mergeMode of mergeModes) {
    it(`No returnState, mergeMode=${BidirectionalMergeMode[mergeMode]}`, () => {
      createLayerAndData(mergeMode, false);
      let y = bidi.apply(x);
      if (mergeMode === null) {
        y = y as Tensor[];
        expect(y.length).toEqual(2);
        expectTensorsClose(
            y[0], tensor2d([[0.9440416, 0.9440416, 0.9440416]], [1, 3]));
        expectTensorsClose(
            y[1], tensor2d([[-0.9842659, -0.9842659, -0.9842659]], [1, 3]));
      } else if (mergeMode === BidirectionalMergeMode.CONCAT) {
        y = y as Tensor;
        expectTensorsClose(
            y,
            tensor2d(
                [[
                  0.9440416, 0.9440416, 0.9440416, -0.9842659, -0.9842659,
                  -0.9842659
                ]],
                [1, 6]));
      } else if (mergeMode === BidirectionalMergeMode.MUL) {
        y = y as Tensor;
        expectTensorsClose(
            y, tensor2d([[-0.929188, -0.929188, -0.929188]], [1, 3]));
      }
    });
  }
  it('returnState', () => {
    createLayerAndData(BidirectionalMergeMode.AVE, true);
    const y = bidi.apply(x) as Tensor[];
    expect(y.length).toEqual(3);
    expectTensorsClose(
        y[0], tensor2d([[-0.02011216, -0.02011216, -0.02011216]], [1, 3]));
    expectTensorsClose(
        y[1], tensor2d([[0.9440416, 0.9440416, 0.9440416]], [1, 3]));
    expectTensorsClose(
        y[2], tensor2d([[-0.9842659, -0.9842659, -0.9842659]], [1, 3]));
  });
});
