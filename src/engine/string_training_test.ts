// Show off VocabLayer when  you get to this point.


import * as tfl from '../index';
import {initializers} from '../index';
import {describeMathCPU, describeMathCPUAndGPU} from '../utils/test_utils';

describeMathCPU('String preproc : Model.predict', () => {
  fit('basic model usage: predict', () => {
    // Define the vocabulary initializer
    const vocabInitializer = initializers.knownVocab(
        {strings: ['hello', 'world', 'こんにちは', '世界']});
    // Define a model with just a vocab layer
    const vocabModel = tfl.sequential({
      layers: [tfl.layers.vocab({
        name: 'myVocabLayer',
        knownVocabSize: 4,
        hashVocabSize: 1,
        vocabInitializer,
        inputShape: [2]  // two words per example
      })]
    });
    console.log(vocabModel.predict(
        tfl.preprocessing.stringTensor2d([['hello', 'world']], [1, 2])));
  });
});

describeMathCPUAndGPU('String preproc : Model.fit', () => {
  it('basic model usage: fit', () => {
    // Define the vocabulary initializer
    const vocabInitializer = initializers.knownVocab(
        {strings: ['hello', 'world', 'こんにちは', '世界']});
    // Define a model with just a vocab layer
    tfl.sequential({
      layers: [tfl.layers.vocab({
        name: 'myVocabLayer',
        knownVocabSize: 4,
        hashVocabSize: 1,
        vocabInitializer,
        inputShape: [2]  // two words per example
      })]
    });
  });
});


// ORIGINAL SKETCH
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
