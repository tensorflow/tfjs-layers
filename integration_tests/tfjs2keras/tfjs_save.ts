/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// import * as tfc from '@tensorflow/tfjs-core';
import * as tf from '@tensorflow/tfjs';
import * as tfjsNode from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import {join} from 'path';

// import * as tfl from '@tensorflow/tfjs-layers';

const tfl = tf;

// TODO(cais): Doc string.
function runAndSaveRandomInputAndOutput(
    model: tf.Model, exportPath: string, inputIntegerMax?: number) {
  // Create a random input and get its predict() output.
  if (model.inputs.length !== 1) {
    throw new Error(
        `Expects model to have exactly 1 input, ` +
        `but got ${model.inputs.length}.`)
  }
  const inputShape = model.inputs[0].shape;
  console.log(`1. inputShape = ${JSON.stringify(inputShape)}`);  // DEBUG
  inputShape[0] = 1;
  console.log(`2. inputShape = ${JSON.stringify(inputShape)}`);  // DEBUG

  const xs = inputIntegerMax == null ?
      tf.randomNormal(inputShape) :
      tf.floor(tf.randomUniform(inputShape, 0, inputIntegerMax));
  if (inputIntegerMax != null) {
    console.log('xs:');  // DEBUG
    xs.print();          // DEBUG
  }
  const ys = model.predict(xs) as tf.Tensor;
  fs.writeFileSync(
      exportPath + '.xs-data.json', JSON.stringify(Array.from(xs.dataSync())));
  fs.writeFileSync(exportPath + '.xs-shape.json', JSON.stringify(xs.shape));
  fs.writeFileSync(
      exportPath + '.ys-data.json', JSON.stringify(Array.from(ys.dataSync())));
  fs.writeFileSync(exportPath + '.ys-shape.json', JSON.stringify(ys.shape));
}

// Multi-layer perceptron (MLP).
async function exportMLPModel(exportPath: string) {
  const model = tfl.sequential();
  // Test both activations encapsulated in other layers and as standalone
  // layers.
  model.add(
      tfl.layers.dense({units: 100, inputShape: [200], activation: 'relu'}));
  model.add(tfl.layers.dense({units: 50, activation: 'elu'}));
  model.add(tfl.layers.dense({units: 24}));
  model.add(tfl.layers.activation({activation: 'elu'}));
  model.add(tfl.layers.dense({units: 8, activation: 'softmax'}));
  await model.save(`file://${exportPath}`);

  runAndSaveRandomInputAndOutput(model, exportPath);
}

// Convolutional neural network (CNN).
async function exportCNNModel(exportPath: string) {
  const model = tfl.sequential();

  // Cover separable and non-separable convoluational layers.
  const inputShape = [40, 40, 3];
  model.add(tfl.layers.conv2d({
    filters: 32,
    kernelSize: [3, 3],
    strides: [2, 2],
    inputShape,
    padding: 'valid',
  }));
  model.add(tfl.layers.batchNormalization({}));
  model.add(tfl.layers.activation({activation: 'relu'}));
  model.add(tfl.layers.dropout({rate: 0.5}));
  model.add(tfl.layers.maxPooling2d({poolSize: 2}));
  model.add(tfl.layers.separableConv2d({
    filters: 32,
    kernelSize: [4, 4],
    strides: [3, 3],
  }));
  model.add(tfl.layers.batchNormalization({}));
  model.add(tfl.layers.activation({activation: 'relu'}));
  model.add(tfl.layers.dropout({rate: 0.5}));
  model.add(tfl.layers.avgPooling2d({poolSize: [2, 2]}));
  model.add(tfl.layers.flatten({}));
  model.add(tfl.layers.dense({units: 100, activation: 'softmax'}));
  await model.save(`file://${exportPath}`);

  runAndSaveRandomInputAndOutput(model, exportPath);
}

async function exportDepthwiseCNNModel(exportPath: string) {
  const model = tfl.sequential();

  // Cover depthwise 2D convoluational layer.
  model.add(tf.layers.depthwiseConv2d({
    depthMultiplier: 2,
    kernelSize: [3, 3],
    strides: [2, 2],
    inputShape: [40, 40, 3],
    padding: 'valid',
  }));
  model.add(tfl.layers.batchNormalization({}));
  model.add(tfl.layers.activation({activation: 'relu'}));
  model.add(tfl.layers.dropout({rate: 0.5}));
  model.add(tfl.layers.maxPooling2d({poolSize: 2}));
  model.add(tfl.layers.flatten({}));
  model.add(tfl.layers.dense({units: 100, activation: 'softmax'}));
  await model.save(`file://${exportPath}`);

  runAndSaveRandomInputAndOutput(model, exportPath);
}

// SimpleRNN with embedding.
async function exportSimpleRNNModel(exportPath: string) {
  const model = tfl.sequential();
  const inputDim = 100;
  model.add(tfl.layers.embedding({inputDim, outputDim: 20, inputShape: [10]}));
  model.add(tfl.layers.simpleRNN({units: 4}));
  await model.save(`file://${exportPath}`);

  runAndSaveRandomInputAndOutput(model, exportPath, inputDim);
}

// GRU with embedding.
async function exportGRUModel(exportPath: string) {
  const model = tfl.sequential();
  const inputDim = 100;
  model.add(tfl.layers.embedding({inputDim, outputDim: 20, inputShape: [10]}));
  model.add(tfl.layers.gru({units: 4, goBackwards: true}));
  await model.save(`file://${exportPath}`);

  runAndSaveRandomInputAndOutput(model, exportPath, inputDim);
}

// Bidirecitonal LSTM with embedding.
async function exportBidirectionalLSTMModel(exportPath: string) {
  const model = tfl.sequential();
  const inputDim = 100;
  model.add(tfl.layers.embedding({inputDim, outputDim: 20, inputShape: [10]}));
  // TODO(cais): Investigate why the `tfl.layers.RNN` typing doesn't work.
  // tslint:disable-next-line:no-any
  const lstm = tfl.layers.lstm({units: 4, goBackwards: true}) as any;
  model.add(tfl.layers.bidirectional({layer: lstm, mergeMode: 'concat'}));
  await model.save(`file://${exportPath}`);

  runAndSaveRandomInputAndOutput(model, exportPath, inputDim);
}

// LSTM + time-distributed layer with embedding.
async function exportTimeDistributedLSTMModel(exportPath: string) {
  const model = tfl.sequential();
  const inputDim = 100;
  model.add(tfl.layers.embedding({inputDim, outputDim: 20, inputShape: [10]}));
  model.add(tfl.layers.lstm({units: 4, returnSequences: true}));
  model.add(tfl.layers.timeDistributed({
    layer: tfl.layers.dense({units: 2, useBias: false, activation: 'softmax'})
  }));
  await model.save(`file://${exportPath}`);

  runAndSaveRandomInputAndOutput(model, exportPath, inputDim);
}

// Model with Conv1D and Pooling1D layers.
async function exportOneDimensionalModel(exportPath: string) {
  const model = tfl.sequential();
  model.add(tfl.layers.conv1d(
      {filters: 16, kernelSize: [4], inputShape: [80, 1], activation: 'relu'}));
  model.add(tfl.layers.maxPooling1d({poolSize: 3}));
  model.add(
      tfl.layers.conv1d({filters: 8, kernelSize: [3], activation: 'relu'}));
  model.add(tfl.layers.avgPooling1d({poolSize: 5}));
  model.add(tfl.layers.flatten());
  await model.save(`file://${exportPath}`);

  runAndSaveRandomInputAndOutput(model, exportPath);
}

// Functional model with two Merge layers.
// function exportFunctionalMergeModel(exportPath: string): void {
//   const input1 = tfl.input({shape: [2, 5]});
//   const input2 = tfl.input({shape: [4, 5]});
//   const input3 = tfl.input({shape: [30]});
//   const reshaped1 = tfl.layers.reshape({targetShape: [10]}).apply(input1) as
//       tf.SymbolicTensor;
//   const reshaped2 = tfl.layers.reshape({targetShape: [20]}).apply(input2) as
//       tf.SymbolicTensor;
//   const dense1 =
//       tfl.layers.dense({units: 5}).apply(reshaped1) as tf.SymbolicTensor;
//   const dense2 =
//       tfl.layers.dense({units: 5}).apply(reshaped2) as tf.SymbolicTensor;
//   const dense3 =
//       tfl.layers.dense({units: 5}).apply(input3) as tf.SymbolicTensor;
//   const avg =
//       tfl.layers.average().apply([dense1, dense2]) as tf.SymbolicTensor;
//   const concat = tfl.layers.concatenate({axis: -1}).apply([avg, dense3]) as
//       tf.SymbolicTensor;
//   const output =
//       tfl.layers.dense({units: 1}).apply(concat) as tf.SymbolicTensor;
//   const model = tfl.model({inputs: [input1, input2, input3], outputs:
//   output}); fs.writeFileSync(exportPath, model.toJSON());
// }

// console.log(`Using tfjs-layers version: ${tfl.version_layers}`);
console.log(`Using tfjs version: ${JSON.stringify(tf.version)}`);
console.log(`Using tfjs-node version: ${tfjsNode.version}`);

if (process.argv.length !== 3) {
  throw new Error('Usage: node tfjs_save.ts <test_data_dir>');
}
const testDataDir = process.argv[2];

(async function() {
  await exportMLPModel(join(testDataDir, 'mlp'));
  await exportCNNModel(join(testDataDir, 'cnn'));
  await exportDepthwiseCNNModel(join(testDataDir, 'depthwise_cnn'));
  await exportSimpleRNNModel(join(testDataDir, 'simple_rnn'));
  await exportGRUModel(join(testDataDir, 'gru'));
  await exportBidirectionalLSTMModel(join(testDataDir, 'bidirectional_lstm'));
  await exportTimeDistributedLSTMModel(
      join(testDataDir, 'time_distributed_lstm'));
  await exportOneDimensionalModel(join(testDataDir, 'one_dimensional'));
  // exportFunctionalMergeModel(join(testDataDir,
  // 'functional_merge.json'));
})();
