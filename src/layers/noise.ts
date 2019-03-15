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
 * TensorFlow.js Layers: Noise Layers.
 */


import {serialization, Tensor, tidy, greaterEqual, randomUniform} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {Layer, LayerArgs} from '../engine/topology';
import {Shape} from '../keras_format/common';
import {Kwargs} from '../types';
import {getExactlyOneTensor} from '../utils/types_utils';

export class GaussianNoise extends Layer {
  /**
   * Apply additive zero-centered Gaussian noise.
   *
   * This is useful to mitigate overfitting
   * (you could see it as a form of random data augmentation).
   * Gaussian Noise (GS) is a natural choice as corruption process
   * for real valued inputs.
   *
   * As it is a regularization layer, it is only active at training time.
   *
   * # Arguments
   *     stddev: float, standard deviation of the noise distribution.
   *
   * # Input shape
   *         Arbitrary. Use the keyword argument `input_shape`
   *         (tuple of integers, does not include the samples axis)
   *         when using this layer as the first layer in a model.
   *
   * # Output shape
   *         Same shape as input.
   */



  static className = 'GaussianNoise';
  readonly stddev: number;

  constructor(stddev: number, args?: LayerArgs) {
    super(args || {});
    this.supportsMasking = true;
    this.stddev = stddev;
  }

  computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
    return inputShape;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    const config = {stddev: this.stddev};
    Object.assign(config, baseConfig);
    return config;
  }

  call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      const noised = () =>
        (K.randomNormal(input.shape, 0., this.stddev).add(input));
      const output =
        K.inTrainPhase(noised, () => input, kwargs.training || false) as
        Tensor;
      return output;
    });
  }
}
serialization.registerClass(GaussianNoise);


export class GaussianDropout extends Layer {

  /**
   * Apply multiplicative 1-centered Gaussian noise.
   *
   * # Arguments
   *     rate: float, drop probability (as with `Dropout`).
   *        The multiplicative noise will have
   *        standard deviation `sqrt(rate / (1 - rate))`.
   *
   * # Input shape
   *     Arbitrary. Use the keyword argument `input_shape`
   *     (tuple of integers, does not include the samples axis)
   *     when using this layer as the first layer in a model.
   *
   * # Output shape
   *     Same shape as input.
   *
   * # References
   *     - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
   *        http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
   *
   */

  static className = 'GaussianDropout';
  readonly rate: number;

  constructor(rate: number, args?: LayerArgs) {
    super(args || {});
    this.supportsMasking = true;
    this.rate = rate;
  }

  computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
    return inputShape;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    const config = {rate: this.rate};
    Object.assign(config, baseConfig);
    return config;
  }

  call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      if (this.rate > 0. && this.rate < 1.) {
        const noised = () => {
          const stddev = Math.sqrt(this.rate / (1.0 - this.rate));
          return K.dot(input, K.randomNormal(input.shape, 1.0, stddev));
        };
        return K.inTrainPhase(noised, () => input, kwargs.training || false);
      }
      return input;
    });
  }
}
serialization.registerClass(GaussianDropout);

export class AlphaDropout extends Layer {
  /**
   * Applies Alpha Dropout to the input.
   *
   * Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
   * to their original values, in order to ensure the self-normalizing property
   * even after this dropout.
   * Alpha Dropout fits well to Scaled Exponential Linear Units
   * by randomly setting activations to the negative saturation value.
   *
   *
   * # Arguments
   *    rate: float, drop probability (as with `Dropout`).
   *        The multiplicative noise will have
   *        standard deviation `sqrt(rate / (1 - rate))`.
   *    noise_shape: A 1-D `Tensor` of type `int32`, representing the
   *         shape for randomly generated keep/drop flags.
   *
   *
   * # Input shape
   *         Arbitrary. Use the keyword argument `input_shape`
   *         (tuple of integers, does not include the samples axis)
   *         when using this layer as the first layer in a model.
   *
   * # Output shape
   *         Same shape as input.
   *
   * # References
   *     - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
   */



  static className = 'AlphaDropout';
  readonly rate: number;
  readonly noiseShape: Shape;

  constructor(rate: number, noiseShape?: Shape, args?: LayerArgs) {
    super(args || {});
    this.supportsMasking = true;
    this.rate = rate;
    this.noiseShape = noiseShape;
  }

  _getNoiseShape(inputs: Tensor | Tensor[]) {
    return this.noiseShape || getExactlyOneTensor(inputs).shape;
  }

  computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
    return inputShape;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    const config = {rate: this.rate};
    Object.assign(config, baseConfig);
    return config;
  }

  call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[] {
    return tidy(() => {
      if (this.rate < 1 && this.rate > 0) {
        const noiseShape = this._getNoiseShape(inputs);

        const droppedInputs = (input = inputs, rate = this.rate) => {

          input = getExactlyOneTensor(input);

          const alpha = 1.6732632423543772848170429916717;
          const scale = 1.0507009873554804934193349852946;

          const alphaP = -alpha * scale;

          let keptIdx = greaterEqual(randomUniform(noiseShape), rate);

          keptIdx = K.cast(keptIdx, 'float32'); // get default dtype?

          // Get affine transformation params
          const a = ((1 - rate) * (1 + rate * alphaP ** 2)) ** -0.5;
          const b = -a * alphaP * rate;

          // Apply mask
          const x = K.dot(input, keptIdx).add(keptIdx.add(-1).mul(alphaP));

          return x.mul(a).add(b);
        };
        return K.inTrainPhase(droppedInputs, () => getExactlyOneTensor(inputs),
          kwargs.training || false);
      }
      return inputs;
    });
  }
}
serialization.registerClass(AlphaDropout);
