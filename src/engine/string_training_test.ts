// Show off VocabLayer when  you get to this point.


import {Tensor, tensor2d} from '@tensorflow/tfjs-core';
import {expectValuesInRange} from '@tensorflow/tfjs-core/dist/test_util';

import * as tfl from '../index';
import {initializers} from '../index';
import {describeMathCPUAndGPU} from '../utils/test_utils';
import {expectTensorsClose} from '../utils/test_utils';

describeMathCPUAndGPU('String preproc : Model.predict', () => {
  it('basic model usage: Sequential predict', () => {
    // Define the vocabulary initializer
    const vocabInitializer = initializers.knownVocab(
        {strings: ['hello', 'world', 'こんにちは', '世界']});
    // Define a model with just a vocab layer
    const knownVocabSize = 4;
    const hashVocabSize = 1;
    const vocabModel = tfl.sequential({
      layers: [tfl.layers.vocab({
        name: 'myVocabLayer',
        knownVocabSize,
        hashVocabSize,
        vocabInitializer,
        inputShape: [2]  // two words per example
      })]
    });
    // Matches known words.
    const x = tfl.preprocessing.stringTensor2d(
        [['world', 'hello'], ['世界', 'こんにちは']], [2, 2]);
    const y = vocabModel.predict(x) as Tensor;
    const yExpected = tensor2d([[1, 0], [3, 2]], [2, 2], 'int32');
    expectTensorsClose(y, yExpected);
    // Handles unknown words.
    const xOutOfVocab = tfl.preprocessing.stringTensor2d(
        [['these', 'words'], ['are', 'out'], ['of', 'vocabulary']], [3, 2]);
    const yOutOfVocab = vocabModel.predict(xOutOfVocab) as Tensor;
    // Out-of-vocab words should hash to buckets after the knownVocab
    expectValuesInRange(
        yOutOfVocab, knownVocabSize, knownVocabSize + hashVocabSize);
  });
  it('basic model usage: Functional predict', () => {
    // Define the vocabulary initializer
    const vocabInitializer = initializers.knownVocab(
        {strings: ['hello', 'world', 'こんにちは', '世界']});
    // Define a model with just a vocab layer
    const knownVocabSize = 4;
    const hashVocabSize = 1;
    const input = tfl.input({shape: [2]});
    const vocabLayer = tfl.layers.vocab({
      name: 'myVocabLayer',
      knownVocabSize,
      hashVocabSize,
      vocabInitializer,
      inputShape: [2]  // two words per example
    });
    const outputSymbolic = vocabLayer.apply(input) as tfl.SymbolicTensor;
    const vocabModel = tfl.model({inputs: input, outputs: outputSymbolic});
    // Matches known words.
    const x = tfl.preprocessing.stringTensor2d(
        [['world', 'hello'], ['世界', 'こんにちは']], [2, 2]);
    const y = vocabModel.predict(x) as Tensor;
    const yExpected = tensor2d([[1, 0], [3, 2]], [2, 2], 'int32');
    expectTensorsClose(y, yExpected);
    // Handles unknown words.
    const xOutOfVocab = tfl.preprocessing.stringTensor2d(
        [['these', 'words'], ['are', 'out'], ['of', 'vocabulary']], [3, 2]);
    const yOutOfVocab = vocabModel.predict(xOutOfVocab) as Tensor;
    // Out-of-vocab words should hash to buckets after the knownVocab
    expectValuesInRange(
        yOutOfVocab, knownVocabSize, knownVocabSize + hashVocabSize);
  });
});



//  ORIGINAL SKETCH
/*
describeMathCPUAndGPU('String Preproc Model.fit', () => {
  // Define the vocabulary initializer
  const vocabInitializer = initializers.knownVocab([
    "hello", "world", "benkyou", "suru"
  ]);
  // Define a model with just a vocab layer
  const vocabModel = models.sequential([
    preprocessingLayers.vocab({
      name: 'myVocabLayer',
      knownVocabSize: 4,
      hashVocabSize: 1,
      vocabInitializer: vocabInitializer
    })]);
  // Compile the model with an optimizer for the vocab layer.
  vocabModel.compile({
    optimizer: vocabCounter,
    loss: undefined,
  });
  const trainInputs = stringTensor1D([
    "a", "a", "b", "b", "c", "c", "d", "d"]);
  // Fit the model to a tensor of strings
  await vocabModel.fit(
    trainInputs, trainInputs, { batchSize: 1, epochs: 1 });
  // call predict on a string of inputs and expect the new vocab values.
  const testInputs = stringTensor1D([
    "a", "b", "c","d", "hello"]);
  testOutputs = model.predict(testInputs);
  expectArraysClose(testOutputs, [0, 1, 2, 3, 4]);

});
*/
