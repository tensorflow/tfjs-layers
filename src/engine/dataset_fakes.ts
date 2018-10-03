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

import {Shape} from '../types';
import {LazyIterator, TensorOrTensorMap, Dataset,} from './dataset_stub';

export interface FakeDatasetConfig {
  /**
   * The shape(s) of the features of a single example.
   *
   * Use an object mapping name to shape, if more than one feature tensors
   * are required.
   */
  xShape: Shape|{[name: string]: Shape};

  /**
   * The shape of the target(s) of a single exapmle.
   */
  yShape: Shape|{[name: string]: Shape};

  /**
   * A function that generates preset sequence of X tensors.
   *
   * This function is invoked each time a new iterator is created.
   */
  xTensorsFunc?: () => tfc.Tensor[] | {[name: string]: tfc.Tensor[]};

  /**
   * A function that generates preset sequence of Y tensors.
   *
   * This function is invoked each time a new iterator is created.
   */
  yTensorsFunc?: () => tfc.Tensor[] | {[name: string]: tfc.Tensor[]};

  /**
   * The size of each batch generated by the iterator.
   */
  batchSize: number;

  /**
   * The number of batches an iterator generates before declaring done to be
   * true.
   */
  numBatches: number;
}

function mergeBatchSizeAndShape(
    batchSize: number, shape: Shape|{[name: string]: Shape}): Shape|
    {[name: string]: Shape} {
  if (Array.isArray(shape)) {
    return [batchSize].concat(shape);
  } else {
    const output: {[name: string]: Shape} = {};
    for (const name in shape) {
      output[name] = [batchSize].concat(shape[name]);
    }
    return output;
  }
}

function generateRandomTensorContainer(shape: Shape|{[name: string]: Shape}):
    tfc.Tensor|{[name: string]: tfc.Tensor} {
  let output: tfc.Tensor|{[name: string]: tfc.Tensor};
  if (Array.isArray(shape)) {
    output = tfc.randomNormal(shape);
  } else {
    output = {};
    for (const name in shape) {
      output[name] = tfc.randomNormal(shape[name]);
    }
  }
  return output;
}

class FakeNumericIterator extends
    LazyIterator<[TensorOrTensorMap, TensorOrTensorMap]> {
  private xBatchShape: Shape|{[name: string]: Shape};
  private yBatchShape: Shape|{[name: string]: Shape};
  private numBatches: number;
  private batchCount: number;
  private xTensorsFunc: () => tfc.Tensor[] | {[name: string]: tfc.Tensor[]};
  private yTensorsFunc: () => tfc.Tensor[] | {[name: string]: tfc.Tensor[]};
  private xTensorValues: tfc.Tensor[]|{[name: string]: tfc.Tensor[]};
  private yTensorValues: tfc.Tensor[]|{[name: string]: tfc.Tensor[]};
  private tensorIndex = 0;

  constructor(config: FakeDatasetConfig) {
    super();
    this.xBatchShape = mergeBatchSizeAndShape(config.batchSize, config.xShape);
    this.yBatchShape = mergeBatchSizeAndShape(config.batchSize, config.yShape);
    this.numBatches = config.numBatches;
    this.batchCount = 0;
    this.xTensorsFunc = config.xTensorsFunc;
    this.yTensorsFunc = config.yTensorsFunc;

    // Sanity check on the preset tensors.
    tfc.util.assert(
        this.xTensorsFunc == null && this.yTensorsFunc == null ||
            this.xTensorsFunc != null && this.yTensorsFunc != null,
        'presetXTensors and presetYTensors must be both null/undefined ' +
            'or both set.');
  }

  async next():
      Promise<IteratorResult<[TensorOrTensorMap, TensorOrTensorMap]>> {
    const done = ++this.batchCount > this.numBatches;
    if (this.xTensorsFunc == null) {
      // Generate data randomly.
      return {
        done,
        value: done ? null :
                      [
                        generateRandomTensorContainer(this.xBatchShape),
                        generateRandomTensorContainer(this.yBatchShape)
                      ]
      };
    } else {
      // Use preset tensors.
      if ((this.batchCount - 1) % this.numBatches === 0) {
        this.xTensorValues = this.xTensorsFunc();
        this.yTensorValues = this.yTensorsFunc();
        this.tensorIndex = 0;
      }
      const index = this.tensorIndex++;

      let xs: tfc.Tensor|{[name: string]: tfc.Tensor};
      if (Array.isArray(this.xTensorValues)) {
        xs = (this.xTensorValues as tfc.Tensor[])[index];
        tfc.util.assert(
            tfc.util.arraysEqual(xs.shape, this.xBatchShape as Shape),
            `Shape mismatch: expected: ${JSON.stringify(this.xBatchShape)}; ` +
                `actual: ${JSON.stringify(xs.shape)}`);
      } else {
        xs = {};
        for (const key in this.xTensorValues) {
          xs[key] = this.xTensorValues[key][index];
          tfc.util.assert(
              tfc.util.arraysEqual(xs[key].shape, this.xBatchShape as Shape),
              `Shape mismatch: expected: ${
                  JSON.stringify(this.xBatchShape)}; ` +
                  `actual: ${JSON.stringify(xs.shape)}`);
        }
      }

      // TODO(cais): Take care of the case of multiple outputs.
      const ys =
          (this.yTensorValues as tfc.Tensor[])[index] as tfc.Tensor;
      tfc.util.assert(
          tfc.util.arraysEqual(ys.shape, this.yBatchShape as Shape),
          `Shape mismatch: expected: ${JSON.stringify(this.yBatchShape)}; ` +
              `actual: ${JSON.stringify(ys.shape)}`);
      return {done, value: done ? null : [xs, ys]};
    }
  }
}

/**
 * A fake dataset with configurable feature and target shapes.
 *
 * The batch size and # of batches are also configurable.
 *
 * The iterator from the dataset always generate random-normal float32 values.
 */
export class FakeNumericDataset extends
    Dataset<[TensorOrTensorMap, TensorOrTensorMap]> {
  constructor(readonly config: FakeDatasetConfig) {
    super();
    tfc.util.assert(
        config.batchSize > 0 && Number.isInteger(config.batchSize),
        `batchSize must be a positive integer, but got ${config.batchSize}`);
    tfc.util.assert(
        config.numBatches > 0 && Number.isInteger(config.numBatches),
        `numBatches must be positive integer, but got ${config.numBatches}`);
  }

  async iterator():
      Promise<LazyIterator<[TensorOrTensorMap, TensorOrTensorMap]>> {
    return new FakeNumericIterator(this.config);
  }
}
