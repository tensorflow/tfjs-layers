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
import {NotImplementedError, ValueError} from '../errors';
import {disposeTensorsInLogs, UnresolvedLogs} from '../logs';

import {Dataset, TensorMap, TensorOrTensorMap} from './dataset_stub';

/**
 * Interface configuration model training based on data as a dataset object.
 */
export interface ModelFitDatasetConfig<T extends TensorContainer> {
  /**
   * Total number of steps (batches of samples) before
   * declaring one epoch finished and starting the next epoch. It should
   * typically be equal to th enumber of samples of your dataset divided by
   * the batch size, so that `fitDataset`() call can utilize the entire dataset.
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
   * Optional batch size for validation.
   *
   * Used only if `validationData` is an array of `Tensor` objects, i.e., not
   * a dataset object.
   */
  batchSize?: number;

  /**
   * Only relevant if `stepsPerEpoch` is specified.
   *
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

/**
 * Standardize the output of a dataset iterator of Model.fitDataset() use.
 *
 * @param model: A `Model` object.
 * @param iteratorOut The output of a dataset iterator. It is required to be
 *   an array of two tensor containers. Each of the two elements of the array
 *   must be a single `Tensor` or a map from string names to `Tensor`s.
 * @returns A flat array of `Tensor` objects: the input `Tensor`s followed
 *   by the target `Tensor`s.
 */
function standardizeDataIteratorOutput(
    // Type `model` as `any` here to avoid circular dependency w/ training.ts.
    // tslint:disable-next-line:no-any
    model: any, iteratorOut: TensorContainer): tfc.Tensor[] {
  if (model.outputs.length > 1) {
    throw new NotImplementedError(
        `Support for training a model with multiple output tensors with ` +
        `a dataset object is not implemented yet.`);
  }

  tfc.util.assert(
      Array.isArray(iteratorOut) && iteratorOut.length === 2,
      'Dataset iterator for fitDataset() is expected to generate ' +
          'an Array of length 2: `[xs, ys]`, but instead generates ' +
          iteratorOut);
  // TODO(cais): If there are multiple inputs or outputs, make sure
  //   they all have the same batch size.
  iteratorOut = iteratorOut as [TensorOrTensorMap, TensorOrTensorMap];
  const ys = iteratorOut[1] as tfc.Tensor;
  let xs = iteratorOut[0] as TensorOrTensorMap;
  if (xs instanceof tfc.Tensor) {
    tfc.util.assert(
        model.inputs.length === 1,
        `Model has multiple ${model.inputs.length} inputs, hence it ` +
            `expects the input dataset to generate a dictionary of tensors ` +
            ` (with keys ${JSON.stringify(model.inputNames)}, ` +
            `but received a single tensor.`);
    return [xs, ys];
  } else {
    xs = xs as TensorMap;
    const flattendXs: tfc.Tensor[] = [];
    // Check that all the required keys are available and all the batch sizes
    // are equal.
    for (const inputName of model.inputNames) {
      if (xs[inputName] == null) {
        throw new ValueError(
            `The feature data generated by the dataset lacks the required ` +
            `input key '${inputName}'.`);
      }
      flattendXs.push(xs[inputName]);
    }
    return flattendXs.concat(ys);
  }

  // TODO(cais): Handle case in which ys is a TensorMap.
}

function standardizeValidationData<T extends TensorContainer>(
    data:
        [
          tfc.Tensor|tfc.Tensor[], tfc.Tensor|tfc.Tensor[]
        ]|[tfc.Tensor | tfc.Tensor[], tfc.Tensor | tfc.Tensor[],
           tfc.Tensor | tfc.Tensor[]]|
    Dataset<T>): {xs: tfc.Tensor|tfc.Tensor[], ys: tfc.Tensor|tfc.Tensor[]} {
  if (!Array.isArray(data)) {
    throw new NotImplementedError(
        'Validation with dataset is not implemented yet.');
  } else {
    if (data.length === 3) {
      throw new NotImplementedError(
          'Validation with sample weights is not implemented yet.');
    }
  }
  return {xs: data[0], ys: data[1]};
}

export async function fitDataset<T extends TensorContainer>(
    // Type `model` as `any` here to avoid circular dependency w/ training.ts.
    // tslint:disable-next-line:no-any
    model: any, dataset: Dataset<T>,
    config: ModelFitDatasetConfig<T>): Promise<History> {
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
    let valXs: tfc.Tensor|tfc.Tensor[];
    let valYs: tfc.Tensor|tfc.Tensor[];
    if (doValidation) {
      const validationData = standardizeValidationData(config.validationData);
      valXs = validationData.xs;
      valYs = validationData.ys;
    }

    const trainFunction = model.makeTrainFunction();
    const outLabels = model.getDedupedMetricsNames() as string[];

    let callbackMetrics: string[];
    if (doValidation) {
      callbackMetrics =
          outLabels.slice().concat(outLabels.map(n => 'val_' + n));
    } else {
      callbackMetrics = outLabels.slice();
    }

    const callbacks = standardizeCallbacks(config.callbacks);
    const {callbackList, history} = configureCallbacks(
        callbacks, config.yieldEvery, config.verbose, config.epochs, null, null,
        config.stepsPerEpoch,
        null,  // Batch size determined by the dataset itself.
        doValidation, callbackMetrics);
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

        const xsAndYs = standardizeDataIteratorOutput(model, iteratorOut.value);
        const batchLogs: UnresolvedLogs = {};
        batchLogs['batch'] = batchIndex;
        batchLogs['size'] = xsAndYs[0].shape[0];

        callbackList.onBatchBegin(batchIndex, batchLogs);

        // Train on batch.
        // TODO(cais): Take care of models with multiple outputs.
        const outs = trainFunction(xsAndYs);
        tfc.dispose(xsAndYs);
        for (let i = 0; i < outLabels.length; ++i) {
          const label = outLabels[i];
          const out = outs[i];
          batchLogs[label] = out;
          tfc.keep(out);
        }

        await callbackList.onBatchEnd(batchIndex, batchLogs);
        disposeTensorsInLogs(batchLogs);

        batchIndex++;
        stepsDone++;

        // Epoch finished. Perform validation.
        if (stepsDone >= config.stepsPerEpoch && doValidation) {
          // TODO(cais): Implement validation based on dataset once
          //   evaluateDataset is implemented.
          const valOuts = model.evaluate(valXs, valYs, {
            batchSize: config.batchSize == null ? 32 : config.batchSize,
            verbose: 0
          });
          for (let i = 0; i < model.metricsNames.length; ++i) {
            epochLogs[`val_${model.metricsNames[i]}`] = valOuts[i];
          }
        }
        if (model.stopTraining_) {
          break;
        }
      }
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
