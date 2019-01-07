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

import {elu, leakyRelu, relu, serialization, Tensor, clipByValue, Variable} from '@tensorflow/tfjs-core';

import {Softmax as softmaxActivation} from '../activations';
import {cast} from '../backend/tfjs_backend';
import {Constraint, getConstraint} from '../constraints';
import {Layer, LayerConfig} from '../engine/topology';
import {getScalar} from '../backend/state';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer} from '../initializers';
import {getRegularizer, Regularizer} from '../regularizers';
import {Kwargs, Shape} from '../types';
import {getExactlyOneTensor, getExactlyOneShape} from '../utils/types_utils';
import {LayerVariable} from '../variables';

export interface ReLULayerConfig extends LayerConfig {
  /**
   * Float, the maximum output value.
   */
  maxValue?: number;
}

/**
 * Rectified Linear Unit activation function.
 *
 * Input shape:
 *   Arbitrary. Use the config field `inputShape` (Array of integers, does
 *   not include the sample axis) when using this layer as the first layer
 *   in a model.
 *
 * Output shape:
 *   Same shape as the input.
 */
export class ReLU extends Layer {
  static className = 'ReLU';
  maxValue: number;

  constructor(config?: ReLULayerConfig) {
    super(config == null ? {} : config);
    this.supportsMasking = true;
    if (config != null) {
      this.maxValue = config.maxValue;
    }
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    inputs = getExactlyOneTensor(inputs);
    let output = relu(inputs);
    if (this.maxValue != null) {
      output = clipByValue(output, 0, this.maxValue);
    }
    return output;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {maxValue: this.maxValue};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(ReLU);

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
    const x = getExactlyOneTensor(inputs);
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
serialization.registerClass(LeakyReLU);

export interface PReLULayerConfig extends LayerConfig {
  /**
   * Initializer for the learnable alpha.
   */
  alphaInitializer?: Initializer;

  /**
   * Regularizer for the learnable alpha.
   */
  alphaRegularizer?: Regularizer;

  /**
   * Constraint for the learnable alpha.
   */
  alphaConstraint?: Constraint;

  /**
   * The axes along which to share learnable parameters for the activation
   * function. For example, if the incoming feature maps are from a 2D
   * convolution with output shape `[numExamples, height, width, channels]`,
   * and you wish to share parameters across space (height and width) so that
   * each filter channels has only one set of parameters, set
   * `shared_axes: =[1, 2]`.
   */
  sharedAxes?: number|number[];
}

export class PReLU extends Layer {
  static className = 'PReLU';
  private readonly alphaInitializer: Initializer;
  private readonly alphaRegularizer: Regularizer;
  private readonly alphaConstraint: Constraint;
  private readonly sharedAxes: number[];
  private paramBoradcast: boolean;
  private alpha: LayerVariable;

  constructor(config?: PReLULayerConfig) {
    super(config);
    if (config == null) {
      config = {};
    }

    this.supportsMasking = true;
    this.alphaInitializer = getInitializer(config.alphaInitializer);
    this.alphaRegularizer = getRegularizer(config.alphaRegularizer);
    this.alphaConstraint = getConstraint(config.alphaConstraint);
    if (config.sharedAxes == null) {
      this.sharedAxes = null;
    } else if (Array.isArray(config.sharedAxes)) {
      this.sharedAxes = config.sharedAxes;
    } else if (typeof config.sharedAxes === 'number') {
      this.sharedAxes = [config.sharedAxes];
    } else {
      throw new ValueError(
          `Expected sharedAxes to be a number or an array of numbers, ` +
          `but got ${config.sharedAxes}`);
    }
  }

  build(inputShape: Shape|Shape[]) {
    inputShape = getExactlyOneShape(inputShape);
    const paramShape: Shape = inputShape.slice(1);
    if (this.sharedAxes != null) {
      for (const i of this.sharedAxes) {
        paramShape[i - 1] = 1;
        this.paramBoradcast = true;
      }
    }
    this.alpha = this.addWeight('alpha', paramShape, 'float32', this.alphaInitializer,
        this.alphaRegularizer, true, this.alphaConstraint);
  }
}

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
    const x = getExactlyOneTensor(inputs);
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
serialization.registerClass(ELU);

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
    const x = getExactlyOneTensor(inputs);
    return x.mul(cast(x.greater(this.thetaTensor), 'float32'));
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
serialization.registerClass(ThresholdedReLU);

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
  readonly softmax: (t: Tensor, a?: number) => Tensor;
  readonly DEFAULT_AXIS = 1.0;

  constructor(config?: SoftmaxLayerConfig) {
    super(config == null ? {} : config);
    if (config == null) {
      config = {};
    }
    this.softmax = new softmaxActivation().apply;
    this.axis = config.axis == null ? this.DEFAULT_AXIS : config.axis;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const x = getExactlyOneTensor(inputs);
    return this.softmax(x, this.axis);
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
serialization.registerClass(Softmax);
