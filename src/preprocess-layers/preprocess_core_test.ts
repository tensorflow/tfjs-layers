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
import {eye, Tensor, tensor1d, tensor2d, zeros} from '@tensorflow/tfjs-core';
import {expectArraysEqual, expectValuesInRange} from '@tensorflow/tfjs-core/dist/test_util';

import * as tfl from '../index';
import {describeMathCPU, describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from '../utils/test_utils';

import {VocabLayer} from './preprocess_core';

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

describeMathCPU('Vocab Layer: Symbolic', () => {
  const symbolicInputs = [
    new tfl.SymbolicTensor('string', [10, 4], null, [], null),
    new tfl.SymbolicTensor('string', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('string', [14, 12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('string', [null, 4], null, [], null),
    new tfl.SymbolicTensor('string', [null, 10, 4], null, [], null),
    new tfl.SymbolicTensor('string', [null, 12, 10, 4], null, [], null),
  ];

  for (const symbolicInput of symbolicInputs) {
    it(`Generates correct symbolic output: ` +
           `input shape=${JSON.stringify(symbolicInput.shape)}`,
       () => {
         const hashVocabSize = 0;
         const knownVocabSize = 10000;
         const vocabLayer = tfl.layers.vocab({hashVocabSize, knownVocabSize});
         const output = vocabLayer.apply(symbolicInput) as tfl.SymbolicTensor;

         const expectedShape = symbolicInput.shape;
         expect(output.shape).toEqual(expectedShape);
         expect(output.sourceLayer).toEqual(vocabLayer);
         expect(output.inputs).toEqual([symbolicInput]);
       });
  }

  it('Vocab connects to dense: With undetermined batch dimension', () => {
    const input1 = new tfl.SymbolicTensor('string', [null, 4], null, [], null);
    const vocabLayer1 = tfl.layers.vocab({knownVocabSize: 10000});
    const output1 = vocabLayer1.apply(input1) as tfl.SymbolicTensor;

    const denseLayer2 = tfl.layers.dense({units: 6});
    const output2 = denseLayer2.apply(output1) as tfl.SymbolicTensor;

    expect(output1.shape).toEqual([null, 4]);
    expect(output1.sourceLayer).toEqual(vocabLayer1);
    expect(output1.inputs).toEqual([input1]);
    expect(output2.shape).toEqual([null, 6]);
    expect(output2.sourceLayer).toEqual(denseLayer2);
    expect(output2.inputs).toEqual([output1]);
  });
});

describeMathCPUAndGPU('Vocab Layer: Tensor', () => {
  const knownVocabSize = 100;
  it('Call with known tokens', () => {
    const inputStrings =
        tfl.preprocessing.stringTensor2d([['hello'], ['world']], [2, 1]);
    const vocabLayer =
        tfl.layers.vocab({knownVocabSize, hashVocabSize: 0}) as VocabLayer;
    const myVocab = new Map<string, number>([['hello', 0], ['world', 1]]);
    vocabLayer.setVocab(myVocab);
    const expectedOutput = tensor2d([[0], [1]], [2, 1], 'int32');
    expectTensorsClose(
        vocabLayer.apply(inputStrings, null) as Tensor, expectedOutput);
  });


  function randString() {
    return Math.random().toString(36).substr(2, length + 2);
  }

  const hashVocabSizes: number[] = [1, 3, 100];
  for (const hashVocabSize of hashVocabSizes) {
    it(`Call with unknown tokens to test hash ${hashVocabSize}`, () => {
      const rVec = [randString(), randString(), randString(), randString()];
      const inputStrings = tfl.preprocessing.stringTensor1d(rVec);
      const vocabLayer =
          tfl.layers.vocab({knownVocabSize, hashVocabSize}) as VocabLayer;
      const myVocab = new Map<string, number>([['hello', 0], ['world', 1]]);
      vocabLayer.setVocab(myVocab);
      const outputTensor = vocabLayer.apply(inputStrings, null) as Tensor;
      expectArraysEqual(outputTensor.shape, [4]);
      expectValuesInRange(
          outputTensor, knownVocabSize, knownVocabSize + hashVocabSize - 1);
    });
  }

  it('Unknown token but no hashVocbSize throws error', () => {
    const rVec = [randString(), randString(), randString(), randString()];
    const inputStrings = tfl.preprocessing.stringTensor1d(rVec);
    const vocabLayer =
        tfl.layers.vocab({knownVocabSize, hashVocabSize: 0}) as VocabLayer;
    const myVocab = new Map<string, number>([['hello', 0], ['world', 1]]);
    vocabLayer.setVocab(myVocab);
    expect(() => vocabLayer.apply(inputStrings, null))
        .toThrowError(/Key not in vocab/);
  });

  // DOING DOING DOING.  Make it possible to save & load a vocabulary layer
  // using the serialization api.

  // DOING DOING DOING.  Make model.fit work with VocabLayer.
});
