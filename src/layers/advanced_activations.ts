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
 *  Advanced activation layers.
 */

import {elu, leakyRelu, serialization, Tensor} from '@tensorflow/tfjs-core';

import {softmax} from '../activations';
import {cast} from '../backend/tfjs_backend';
import {getScalar} from '../backend/tfjs_backend';
import {Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError} from '../errors';
import {DType, Kwargs, Shape} from '../types';
import * as generic_utils from '../utils/generic_utils';

export interface LeakyReLULayerConfig extends LayerConfig {
  /**
   * Float `>= 0`. Negative slope coefficient. Defaults to `0.3`.
   */
  alpha?: number;
}

/**
 * Leaky version of a rectified linear unit.
 *
 * It allows a small gradient when the unit is not active:
 * `f(x) = alpha * x for x < 0.`
 * `f(x) = x for x >= 0.`
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 */
export class LeakyReLU extends Layer {
  static className = 'LeakyReLU';
  readonly alpha: number;

  readonly DEFAULT_ALPHA = 0.3;

  constructor(config?: LeakyReLULayerConfig) {
    super(config == null ? {} : config);
    if (config == null) {
      config = {};
    }

    this.alpha = config.alpha == null ? this.DEFAULT_ALPHA : config.alpha;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const x = generic_utils.getExactlyOneTensor(inputs);
    return leakyRelu(x, this.alpha);
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {alpha: this.alpha};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(LeakyReLU);

// TODO(cais): Implement PReLU

export interface ELULayerConfig extends LayerConfig {
  /**
   * Float `>= 0`. Negative slope coefficient. Defaults to `1.0`.
   */
  alpha?: number;
}

/**
 * Exponetial Linear Unit (ELU).
 *
 * It follows:
 * `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
 * `f(x) = x for x >= 0`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Fast and Accurate Deep Network Learning by Exponential Linear Units
 * (ELUs)](https://arxiv.org/abs/1511.07289v1)
 */
export class ELU extends Layer {
  static className = 'ELU';
  readonly alpha: number;

  readonly DEFAULT_ALPHA = 1.0;

  constructor(config?: ELULayerConfig) {
    super(config == null ? {} : config);
    if (config == null) {
      config = {};
    }

    if (config.alpha != null && config.alpha !== this.DEFAULT_ALPHA) {
      throw new NotImplementedError(
          `Non-default alpha value (${config.alpha}) is not supported by the ` +
          `ELU layer yet.`);
    }

    this.alpha = config.alpha == null ? this.DEFAULT_ALPHA : config.alpha;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const x = generic_utils.getExactlyOneTensor(inputs);
    return elu(x);
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {alpha: this.alpha};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(ELU);

export interface ThresholdedReLULayerConfig extends LayerConfig {
  /**
   * Float >= 0. Threshold location of activation.
   */
  theta?: number;
}

/**
 * Thresholded Rectified Linear Unit.
 *
 * It follows:
 * `f(x) = x for x > theta`,
 * `f(x) = 0 otherwise`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Zero-Bias Autoencoders and the Benefits of Co-Adapting
 * Features](http://arxiv.org/abs/1402.3337)
 */
export class ThresholdedReLU extends Layer {
  static className = 'ThresholdedReLU';
  readonly theta: number;
  private readonly thetaTensor: Tensor;

  readonly DEFAULT_THETA = 1.0;

  constructor(config?: ThresholdedReLULayerConfig) {
    super(config == null ? {} : config);
    if (config == null) {
      config = {};
    }

    this.theta = config.theta == null ? this.DEFAULT_THETA : config.theta;
    this.thetaTensor = getScalar(this.theta);
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const x = generic_utils.getExactlyOneTensor(inputs);
    return x.mul(cast(x.greater(this.thetaTensor), DType.float32));
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {theta: this.theta};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(ThresholdedReLU);

export interface SoftmaxLayerConfig extends LayerConfig {
  /**
   * Integer, axis along which the softmax normalization is applied.
   * Defaults to `-1` (i.e., the last axis).
   */
  axis?: number;
}

/**
 * Softmax activation layer.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 */
export class Softmax extends Layer {
  static className = 'Softmax';
  readonly axis: number;

  readonly DEFAULT_AXIS = 1.0;

  constructor(config?: ThresholdedReLULayerConfig) {
    super(config == null ? {} : config);
    if (config == null) {
      config = {};
    }

    this.axis = config.theta == null ? this.DEFAULT_AXIS : config.theta;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const x = generic_utils.getExactlyOneTensor(inputs);
    return softmax(x, this.axis);
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {axis: this.axis};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(Softmax);
