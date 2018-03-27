/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as ui from './ui';

async function runBenchmark(artifactsDir, modelName, config) {
  const modelPath = artifactsDir + modelName + '/';
  console.log('Loading model "' + modelName + '" and benchmark data...');
  const model = await tf.loadModel(modelPath + 'model.json');
  console.log('Done loading model "' + modelName + '" and benchmark data.');

  const benchmarkData = await (await fetch(modelPath + 'data.json')).json();

  const lossMap = {
    mean_squared_error: 'meanSquaredError',
    categorical_crossentropy: 'categoricalCrossentropy',
  };
  // TODO(cais): Maybe TF.js Layers should tolerate these Python-style names
  // for losses.

  const batchSize = benchmarkData.batch_size;
  const xs = tf.randomUniform(
      [batchSize].concat(benchmarkData.input_shape));
  const ys = tf.randomUniform(
      [batchSize].concat(benchmarkData.target_shape));
  model.compile({
    optimizer: benchmarkData.optimizer,
    loss: lossMap[benchmarkData.loss],
  });

  const FIT_BURNIN_EPOCHS = config.FIT_BURNIN_EPOCHS;
  const PREDICT_BURNINS = config.PREDICT_BURNINS;
  const PREDICT_RUNS = config.PREDICT_RUNS;

  // Perform fit() burn-in.
  await model.fit(
      xs, ys, {batchSize: benchmarkData.batch_size, epochs: FIT_BURNIN_EPOCHS});
  model.trainableWeights[0].read().dataSync();

  const trainBeginMs = performance.now();
  await model.fit(xs, ys, {
    batchSize: benchmarkData.batch_size,
    epochs: benchmarkData.train_epochs
  });
  // After the fit() call, call dataSync() to let the scheduled GPU
  // operations to complete before proceeding.
  model.trainableWeights[0].read().dataSync();
  const trainEndMs = performance.now();
  const trainTimeMs = (trainEndMs - trainBeginMs) / benchmarkData.train_epochs;

  // Perform predict() burn-in.
  let output;
  for (let i = 0; i < PREDICT_BURNINS; ++i) {
    output = model.predict(xs);
  }
  // Time predict() a number of times and take the average.
  const predictBeginMs = performance.now();
  for (let i = 0; i < PREDICT_RUNS; ++i) {
    output = model.predict(xs);
  }
  // After all the model.predict() calls, invoke dataSync() once to let the
  // scheduled GPU operations complete before proceeding.
  output.dataSync();
  const predictEndMs = performance.now();
  const predictTimeMs = (predictEndMs - predictBeginMs) / PREDICT_RUNS;
  return {
    originalData: benchmarkData,
    predictTimeMs: predictTimeMs,
    trainTimeMs: trainTimeMs,
  };
}

async function getRunAllBenchmarks(artifactsDir, benchmarks) {
  return async () => {
    ui.status('Running benchmarks...');
    for (let i = 0; i < benchmarks.models.length; ++i) {
      const modelName = benchmarks.models[i];
      ui.status(
          'Running model (' + (i + 1) + ' of ' + benchmarks.models.length +
          '): "' + modelName +
          '" ... (Please wait patiently. Do NOT click anything.)');
      await tf.nextFrame();
      console.log('Benchmarking model: ' + modelName);
      const result =
          await runBenchmark(artifactsDir, modelName, benchmarks.config);
      ui.addResult(modelName, result);
    }
    status('Standing by.');
  };
}

async function setupBenchmarks() {
  const artifactsDir = './data/';

  console.log('Loading benchmarks...');
  const benchmarks =
      await (await fetch(artifactsDir + 'benchmarks.json')).json();
  console.log('Done loading benchmarks:', benchmarks);

  ui.setMetadata(benchmarks.metadata);
  ui.setRunBenchmarksFunction(getRunAllBenchmarks(artifactsDir, benchmarks));
}

setupBenchmarks();
