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


import {serialization, Tensor, tidy} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {Layer, LayerArgs} from '../engine/topology';
import {Shape} from '../keras_format/common';
import {Kwargs} from '../types';
import {getExactlyOneTensor} from '../utils/types_utils';

export class GaussianNoise extends Layer {
  static className = 'GaussianNoise';
  readonly stddev: number;

  constructor(stddev: number, args?: LayerArgs) {
    super(args || {});
    this.supportsMasking = true;
    this.stddev = stddev;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    const config = {stddev: this.stddev};
    Object.assign(config, baseConfig);
    return config;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
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
  static className = 'GaussianDropout';
  readonly rate: number;

  constructor(rate: number, args?: LayerArgs) {
    super(args || {});
    this.supportsMasking = true;
    this.rate = rate;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    const config = {rate: this.rate};
    Object.assign(config, baseConfig);
    return config;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
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
