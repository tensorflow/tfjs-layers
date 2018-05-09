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
 * TensorFlow.js Layers: Pooling Layers.
 */

// tslint:disable:max-line-length
import {max, serialization, squeeze, Tensor} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {checkDataFormat, checkPaddingMode, DataFormat, PaddingMode} from '../common';
import {InputSpec} from '../engine/topology';
import {Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError} from '../errors';
import {Kwargs, Shape} from '../types';
import {convOutputLength} from '../utils/conv_utils';
import * as generic_utils from '../utils/generic_utils';

// tslint:enable:max-line-length

export interface Pooling1DLayerConfig extends LayerConfig {
  /**
   * Size of the window to pool over, should be an integer.
   */
  poolSize?: number;
  /**
   * Period at which to sample the pooled values.
   *
   * If `null`, defaults to `poolSize`.
   */
  strides?: number;
  /** How to fill in data that's not an integer multiple of poolSize. */
  padding?: PaddingMode;
}

/**
 * Abstract class for different pooling 1D layers.
 */
export abstract class Pooling1D extends Layer {
  protected readonly poolSize: [number];
  protected readonly strides: [number];
  protected readonly padding: PaddingMode;

  /**
   *
   * @param config Parameters for the Pooling layer.
   *
   * config.poolSize defaults to 2.
   */
  constructor(config: Pooling1DLayerConfig) {
    if (config.poolSize == null) {
      config.poolSize = 2;
    }
    super(config);
    this.poolSize = [config.poolSize];
    this.strides = config.strides == null ? this.poolSize : [config.strides];
    this.padding = config.padding == null ? 'valid' : config.padding;
    checkPaddingMode(this.padding);
    this.inputSpec = [new InputSpec({ndim: 3})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    length = convOutputLength(
        inputShape[1], this.poolSize[0], this.padding, this.strides[0]);
    return [inputShape[0], length, inputShape[2]];
  }

  protected abstract poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor;

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    this.invokeCallHook(inputs, kwargs);
    // Add dummy last dimension.
    inputs = K.expandDims(generic_utils.getExactlyOneTensor(inputs), 2);
    const output = this.poolingFunction(
        generic_utils.getExactlyOneTensor(inputs), [this.poolSize[0], 1],
        [this.strides[0], 1], this.padding, 'channelsLast');
    // Remove dummy last dimension.
    return squeeze(output, [2]);
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      poolSize: this.poolSize,
      padding: this.padding,
      strides: this.strides,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

/**
 * Max pooling operation for temporal data.
 *
 * Input shape:  `[batchSize, inLength, channels]`
 *
 * Output shape: `[batchSize, pooledLength, channels]`
 */
export class MaxPooling1D extends Pooling1D {
  static className = 'MaxPooling1D';
  constructor(config: Pooling1DLayerConfig) {
    super(config);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return K.pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
  }
}
serialization.SerializationMap.register(MaxPooling1D);

/**
 * Average pooling operation for spatial data.
 *
 * Input shape: `[batchSize, inLength, channels]`
 *
 * Output shape: `[batchSize, pooledLength, channels]`
 *
 * `tf.avgPool1d` is an alias.
 */
export class AveragePooling1D extends Pooling1D {
  static className = 'AveragePooling1D';
  constructor(config: Pooling1DLayerConfig) {
    super(config);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return K.pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
  }
}
serialization.SerializationMap.register(AveragePooling1D);

export interface Pooling2DLayerConfig extends LayerConfig {
  /**
   * Factors by which to downscale in each dimension [vertical, horizontal].
   * Expects an integer or an array of 2 integers.
   *
   * For example, `[2, 2]` will halve the input in both spatial dimension.
   * If only one integer is specified, the same window length
   * will be used for both dimensions.
   */
  poolSize?: number|[number, number];

  /**
   * The size of the stride in each dimension of the pooling window. Expects an
   * integer or an array of 2 integers. Integer, tuple of 2 integers, or None.
   *
   * If `null`, defaults to `poolSize`.
   */
  strides?: [number, number];

  /** The padding type to use for the pooling layer. */
  padding?: PaddingMode;
  /** The data format to use for the pooling layer. */
  dataFormat?: DataFormat;
}

/**
 * Abstract class for different pooling 2D layers.
 */
export abstract class Pooling2D extends Layer {
  protected readonly poolSize: [number, number];
  protected readonly strides: [number, number];
  protected readonly padding: PaddingMode;
  protected readonly dataFormat: DataFormat;

  constructor(config: Pooling2DLayerConfig) {
    if (config.poolSize == null) {
      config.poolSize = [2, 2];
    }
    super(config);
    this.poolSize = Array.isArray(config.poolSize) ?
        config.poolSize :
        [config.poolSize, config.poolSize];
    this.strides = config.strides == null ? this.poolSize : config.strides;
    this.padding = config.padding == null ? 'valid' : config.padding;
    this.dataFormat =
        config.dataFormat == null ? 'channelsLast' : config.dataFormat;
    checkDataFormat(this.dataFormat);
    checkPaddingMode(this.padding);

    this.inputSpec = [new InputSpec({ndim: 4})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    let rows =
        this.dataFormat === 'channelsFirst' ? inputShape[2] : inputShape[1];
    let cols =
        this.dataFormat === 'channelsFirst' ? inputShape[3] : inputShape[2];
    rows =
        convOutputLength(rows, this.poolSize[0], this.padding, this.strides[0]);
    cols =
        convOutputLength(cols, this.poolSize[1], this.padding, this.strides[1]);
    if (this.dataFormat === 'channelsFirst') {
      return [inputShape[0], inputShape[1], rows, cols];
    } else {
      return [inputShape[0], rows, cols, inputShape[3]];
    }
  }

  protected abstract poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor;

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    this.invokeCallHook(inputs, kwargs);
    return this.poolingFunction(
        generic_utils.getExactlyOneTensor(inputs), this.poolSize, this.strides,
        this.padding, this.dataFormat);
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      poolSize: this.poolSize,
      padding: this.padding,
      strides: this.strides,
      dataFormat: this.dataFormat
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

/**
 * Max pooling operation for spatial data.
 *
 * Input shape
 *   - If `dataFormat === CHANNEL_LAST`:
 *       4D tensor with shape:
 *       `[batchSize, rows, cols, channels]`
 *   - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *       `[batchSize, channels, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=CHANNEL_LAST`:
 *       4D tensor with shape:
 *       `[batchSize, pooleRows, pooledCols, channels]`
 *   - If `dataFormat=CHANNEL_FIRST`:
 *       4D tensor with shape:
 *       `[batchSize, channels, pooleRows, pooledCols]`
 */
export class MaxPooling2D extends Pooling2D {
  static className = 'MaxPooling2D';
  constructor(config: Pooling2DLayerConfig) {
    super(config);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return K.pool2d(inputs, poolSize, strides, padding, dataFormat, 'max');
  }
}
serialization.SerializationMap.register(MaxPooling2D);

/**
 * Average pooling operation for spatial data.
 *
 * Input shape:
 *  - If `dataFormat === CHANNEL_LAST`:
 *      4D tensor with shape:
 *      `[batchSize, rows, cols, channels]`
 *  - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *      `[batchSize, channels, rows, cols]`
 *
 * Output shape
 *  - If `dataFormat === CHANNEL_LAST`:
 *      4D tensor with shape:
 *      `[batchSize, pooleRows, pooledCols, channels]`
 *  - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *      `[batchSize, channels, pooleRows, pooledCols]`
 *
 * `tf.avgPool2d` is an alias.
 */
export class AveragePooling2D extends Pooling2D {
  static className = 'AveragePooing2D';
  constructor(config: Pooling2DLayerConfig) {
    super(config);
  }

  protected poolingFunction(
      inputs: Tensor, poolSize: [number, number], strides: [number, number],
      padding: PaddingMode, dataFormat: DataFormat): Tensor {
    checkDataFormat(dataFormat);
    checkPaddingMode(padding);
    return K.pool2d(inputs, poolSize, strides, padding, dataFormat, 'avg');
  }
}
serialization.SerializationMap.register(AveragePooling2D);

/**
 * Abstract class for different global pooling 1D layers.
 */
export abstract class GlobalPooling1D extends Layer {
  constructor(config: LayerConfig) {
    super(config);
    this.inputSpec = [new InputSpec({ndim: 3})];
  }

  computeOutputShape(inputShape: Shape): Shape {
    return [inputShape[0], inputShape[2]];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    throw new NotImplementedError();
  }
}

/**
 * Global average pooling operation for temporal data.
 *
 * Input Shape: 3D tensor with shape: `[batchSize, steps, features]`.
 *
 * Output Shape:2D tensor with shape: `[batchSize, features]`.
 */
export class GlobalAveragePooling1D extends GlobalPooling1D {
  static className = 'GlobalAveragePooling1D';
  constructor(config: LayerConfig) {
    super(config);
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const input = generic_utils.getExactlyOneTensor(inputs);
    return K.mean(input, 1);
  }
}
serialization.SerializationMap.register(GlobalAveragePooling1D);

/**
 * Global max pooling operation for temporal data.
 *
 * Input Shape: 3D tensor with shape: `[batchSize, steps, features]`.
 *
 * Output Shape:2D tensor with shape: `[batchSize, features]`.
 */
export class GlobalMaxPooling1D extends GlobalPooling1D {
  static className = 'GlobalMaxPooling1D';
  constructor(config: LayerConfig) {
    super(config);
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const input = generic_utils.getExactlyOneTensor(inputs);
    return max(input, 1);
  }
}
serialization.SerializationMap.register(GlobalMaxPooling1D);

export interface GlobalPooling2DLayerConfig extends LayerConfig {
  /**
   * One of `CHANNEL_LAST` (default) or `CHANNEL_FIRST`.
   *
   * The ordering of the dimensions in the inputs. `CHANNEL_LAST` corresponds
   * to inputs with shape `[batch, height, width, channels[` while
   * `CHANNEL_FIRST` corresponds to inputs with shape
   * `[batch, channels, height, width]`.
   */
  dataFormat?: DataFormat;
}

/**
 * Abstract class for different global pooling 2D layers.
 */
export abstract class GlobalPooling2D extends Layer {
  protected dataFormat: DataFormat;
  constructor(config: GlobalPooling2DLayerConfig) {
    super(config);
    this.dataFormat =
        config.dataFormat == null ? 'channelsLast' : config.dataFormat;
    checkDataFormat(this.dataFormat);
    this.inputSpec = [new InputSpec({ndim: 4})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = inputShape as Shape;
    if (this.dataFormat === 'channelsLast') {
      return [inputShape[0], inputShape[3]];
    } else {
      return [inputShape[0], inputShape[1]];
    }
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    throw new NotImplementedError();
  }

  getConfig(): serialization.ConfigDict {
    const config = {dataFormat: this.dataFormat};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}

/**
 * Global average pooling operation for spatial data.
 *
 * Input shape:
 *   - If `dataFormat` is `CHANNEL_LAST`:
 *       4D tensor with shape: `[batchSize, rows, cols, channels]`.
 *   - If `dataFormat` is `CHANNEL_FIRST`:
 *       4D tensor with shape: `[batchSize, channels, rows, cols]`.
 *
 * Output shape:
 *   2D tensor with shape: `[batchSize, channels]`.
 */
export class GlobalAveragePooling2D extends GlobalPooling2D {
  static className = 'GlobalAveragePooling2D';
  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const input = generic_utils.getExactlyOneTensor(inputs);
    if (this.dataFormat === 'channelsLast') {
      return K.mean(input, [1, 2]);
    } else {
      return K.mean(input, [2, 3]);
    }
  }
}
serialization.SerializationMap.register(GlobalAveragePooling2D);

/**
 * Global max pooling operation for spatial data.
 *
 * Input shape:
 *   - If `dataFormat` is `CHANNEL_LAST`:
 *       4D tensor with shape: `[batchSize, rows, cols, channels]`.
 *   - If `dataFormat` is `CHANNEL_FIRST`:
 *       4D tensor with shape: `[batchSize, channels, rows, cols]`.
 *
 * Output shape:
 *   2D tensor with shape: `[batchSize, channels]`.
 */
export class GlobalMaxPooling2D extends GlobalPooling2D {
  static className = 'GlobalMaxPooling2D';
  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const input = generic_utils.getExactlyOneTensor(inputs);
    if (this.dataFormat === 'channelsLast') {
      return max(input, [1, 2]);
    } else {
      return max(input, [2, 3]);
    }
  }
}
serialization.SerializationMap.register(GlobalMaxPooling2D);
