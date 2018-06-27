// Show off VocabLayer when  you get to this point.

/*
import {describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';
import { initializers } from '../index';

describeMathCPUAndGPU('String Preproc Model.fit', () => {
  // Define the vocabulary initializer
  const vocabInitializer = initializers.vocabFromIterable([
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
    perLayerOptimizer: {
      myVocabLayer: 'countVocabFrequencyOptimizer'
    }
    optimizer: undefined,
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
