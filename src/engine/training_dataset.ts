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
 * Interfaces and methods for training models using TensorFlow.js datasets.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {TensorContainer} from '@tensorflow/tfjs-core/dist/tensor_types';

import {BaseCallback, configureCallbacks, CustomCallbackConfig, History, ModelLoggingVerbosity, standardizeCallbacks, YieldEveryOptions} from '../base_callbacks';
import {NotImplementedError} from '../errors';
import {disposeTensorsInLogs, UnresolvedLogs} from '../logs';

import {Dataset} from './dataset_stub';

export interface ModelFitDatasetConfig<T extends TensorContainer> {
  /**
   * Total number of steps (batches of samples) before
   * declaring one epoch finished and starting the next epoch. It should
   * typically be equal to th enumber of samples of your dataset divided by
   * the batch size, so that fitDataset() call can utilize the entire dataset.
   */
  stepsPerEpoch: number;

  /**
   * The number of times to iterate over the training dataset.
   *
   * An integer.
   */
  epochs: number;

  /**
   * Verbosity level.
   *
   * Expected to be 0, 1, or 2. Default: 1.
   *
   * 0 - No printed message during fit() call.
   * 1 - In Node.js (tfjs-node), prints the progress bar, together with
   *     real-time updates of loss and metric values and training speed.
   *     In the browser: no action. This is the default.
   * 2 - Not implemented yet.
   */
  verbose?: ModelLoggingVerbosity;

  /**
   * List of callbacks to be called during training.
   * Can consist of one or more of the following fields: `onTrainBegin`,
   * `onTrainEnd`, `onEpochBegin`, `onEpochEnd`, `onBatchBegin`, `onBatchEnd`.
   */
  callbacks?: BaseCallback[]|CustomCallbackConfig|CustomCallbackConfig[];

  /**
   * Data on which to evaluate the loss and any model
   * metrics at the end of each epoch. The model will not be trained on this
   * data. This could be any of the following:
   *
   *   - a tuple [xVal, yVal]
   *   - a tuple [xVal, yVal, valSampleWeights].
   *   - a dataset object for the validation data.
   *
   * The model will not be trained on this data.
   * `validationData` will override `validationSplit`.
   */
  validationData?:
      [
        tfc.Tensor|tfc.Tensor[], tfc.Tensor|tfc.Tensor[]
      ]|[tfc.Tensor | tfc.Tensor[], tfc.Tensor|tfc.Tensor[],
         tfc.Tensor|tfc.Tensor[]]|Dataset<T>;

  /**
   * Only relevant if `stepsPerEpoch` is specified and is a dataset object.
   * Total number of steps (batches of samples) to validate before stopping.
   */
  validationSteps?: number;

  /**
   * Configures the frequency of yielding the main thread to other tasks.
   *
   * In the browser environment, yielding the main thread can improve the
   * responsiveness of the page during training. In the Node.js environment,
   * it can ensure tasks queued in the event loop can be handled in a timely
   * manner.
   *
   * - The value can be one of the following strings:
   *   - 'auto': automatically determine how frequently the yielding happens
   *     by measuring the duration of each batch of training (default).
   *   - 'batch': yield every batch.
   *   - 'epoch': yield every epoch.
   *   - 'never': never yield. (But yielding can still happen through `await
   *      nextFrame()` calls in custom callbacks.)
   */
  yieldEvery?: YieldEveryOptions;

  /**
   * Epoch at which to start training (useful for resuming a previous training
   * run).
   */
  initialEpoch?: number;
}


export async function fitDataset<T extends TensorContainer>(
    // Type `model` as `any` here to avoid circular dependency w/ training.ts.
    // tslint:disable-next-line:no-any
    model: any, dataset: Dataset<T>,
    config: ModelFitDatasetConfig<T>): Promise<History> {
  // console.log(
  //     'Calling this.makeTrainFunction():
  //     this.collectedTrainableWeights:', this.collectedTrainableWeights);
  //     // DEBUG
  tfc.util.assert(
      model.optimizer != null,
      'You must compile a model before training/testing. Use ' +
          'Model.compile(modelCompileConfig).');

  tfc.util.assert(
      config != null,
      `For fitDataset(), the 2nd argument (config) is required, ` +
          `but it is not provided in this call.`);
  tfc.util.assert(
      config.epochs != null && config.epochs > 0 &&
          Number.isInteger(config.epochs),
      `For fitDataset(), config.epochs is expected to be a positive ` +
          `integer, but got ${config.epochs}`);
  tfc.util.assert(
      config.stepsPerEpoch != null && config.stepsPerEpoch > 0 &&
          Number.isInteger(config.stepsPerEpoch),
      `For fitDataset(), config.stepsPerEpoch is expected to be a ` +
          `positive integer, but got ${config.stepsPerEpoch}`);

  if (model.isTraining) {
    throw new Error(
        'Cannot start training because another fit() call is ongoing.');
  }
  model.isTraining = true;

  try {
    const doValidation = config.validationData != null;
    if (doValidation) {
      throw new NotImplementedError(
          'Support for validation is not implement for fitDataset yet.');
    }

    const trainFunction = model.makeTrainFunction();
    const outLabels = model.getDedupedMetricsNames();

    let callbackMetrics: string[];
    if (doValidation) {
      throw new NotImplementedError('TODO(cais): Implement validatoin');
    } else {
      // valFunction = null;
      // valIns = [];
      callbackMetrics = outLabels.slice();
    }

    const callbacks = standardizeCallbacks(config.callbacks);
    const {callbackList, history} = configureCallbacks(
        callbacks, config.yieldEvery, config.verbose, config.epochs, null, null,
        config.stepsPerEpoch,
        null,  // Batch size determined by the dataset itself.
        doValidation, callbackMetrics);
    // console.log('fitDataset(): callbackList = ', callbackList);  // DEBUG
    model.history = history;

    await callbackList.onTrainBegin();
    let epoch = config.initialEpoch == null ? 0 : config.initialEpoch;
    const epochLogs: UnresolvedLogs = {};
    const dataIterator = await dataset.iterator();
    while (epoch < config.epochs) {
      await callbackList.onEpochBegin(epoch);
      let stepsDone = 0;
      let batchIndex = 0;
      while (stepsDone < config.stepsPerEpoch) {
        const iteratorOut = await dataIterator.next();
        if (iteratorOut.done) {
          console.warn(
              'Your dataset iterator ran out of data; ' +
              'interrupting training. Make sure that your ' +
              'dataset can generate at least `stepsPerEpoch * epochs` ' +
              'batches (in this case, ' +
              `${config.stepsPerEpoch * config.epochs} batches). ` +
              'You may need to use the repeat() function when building ' +
              'your dataset.');
          break;
        }

        // TOOD(cais): Check iteratorOut.done.
        // console.log('*** iteratorOut = ', iteratorOut);  // DEBUG
        const xsAndYs = model.checkDataIteratorOutput(iteratorOut.value);
        const batchLogs: UnresolvedLogs = {};
        batchLogs['batch'] = batchIndex;
        batchLogs['size'] = xsAndYs[0].shape[0];
        // console.log(`batchLogs = ${JSON.stringify(batchLogs)}`);  // DEBUG

        callbackList.onBatchBegin(batchIndex, batchLogs);

        // Train on batch.
        // TODO(cais): Take care of multiple inputs and multiple outputs.
        const outs = trainFunction(xsAndYs);
        tfc.dispose(xsAndYs);
        // for (let i = 0; i < this.metricsNames.length; ++i) {
        //   batchLogs[this.metricsNames[i]] = outs[i];
        // }
        for (let i = 0; i < outLabels.length; ++i) {
          const label = outLabels[i];
          const out = outs[i];
          // console.log(`label = ${label}, out = `, out);  // DEBUG
          batchLogs[label] = out;
          tfc.keep(out);
        }

        await callbackList.onBatchEnd(batchIndex, batchLogs);
        disposeTensorsInLogs(batchLogs);

        batchIndex++;
        stepsDone++;
        if (model.stopTraining_) {
          break;
        }
      }
      // console.log('Calling onEpochEnd: epoch = ', epoch);  // DEBUG
      await callbackList.onEpochEnd(epoch, epochLogs);
      epoch++;
      if (model.stopTraining_) {
        break;
      }
    }
    await callbackList.onTrainEnd();
    await model.history.syncData();
    return model.history;
  } finally {
    model.isTraining = false;
  }
}
