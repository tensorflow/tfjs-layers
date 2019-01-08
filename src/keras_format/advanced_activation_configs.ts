/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ConstraintIdentifier} from '../constraints';
import {InitializerIdentifier} from '../initializers';
import {RegularizerIdentifier} from '../regularizers';

import {ConstraintConfig} from './constraint_config';
import {InitializerConfig} from './initializer_config';
import {RegularizerConfig} from './regularizer_config';
import {LayerBaseConfig, LayerConfig} from './topology_config';

export interface ReLULayerBaseConfig extends LayerBaseConfig {
  /**
   * Float, the maximum output value.
   */
  maxValue?: number;
}

export type ReLULayerConfig = ReLULayerBaseConfig;

export interface LeakyReLULayerBaseConfig extends LayerBaseConfig {
  /**
   * Float `>= 0`. Negative slope coefficient. Defaults to `0.3`.
   */
  alpha?: number;
}

export type LeakyReLULayerConfig = LeakyReLULayerBaseConfig;

export interface PReLULayerBaseConfig extends LayerBaseConfig {
  /**
   * The axes along which to share learnable parameters for the activation
   * function. For example, if the incoming feature maps are from a 2D
   * convolution with output shape `[numExamples, height, width, channels]`,
   * and you wish to share parameters across space (height and width) so that
   * each filter channels has only one set of parameters, set
   * `shared_axes: [1, 2]`.
   */
  sharedAxes?: number|number[];
}

export interface PReLULayerConfig extends LayerConfig {
  /**
   * Initializer for the learnable alpha.
   */
  alphaInitializer?: InitializerConfig|InitializerIdentifier;

  /**
   * Regularizer for the learnable alpha.
   */
  alphaRegularizer?: RegularizerConfig|RegularizerIdentifier;

  /**
   * Constraint for the learnable alpha.
   */
  alphaConstraint?: ConstraintConfig|ConstraintIdentifier;
}

export interface ELULayerBaseConfig extends LayerBaseConfig {
  /**
   * Float `>= 0`. Negative slope coefficient. Defaults to `1.0`.
   */
  alpha?: number;
}

export type ELULayerConfig = ELULayerBaseConfig;

export interface ThresholdedReLULayerBaseConfig extends LayerBaseConfig {
  /**
   * Float >= 0. Threshold location of activation.
   */
  theta?: number;
}

export type ThresholdedReLULayerConfig = ThresholdedReLULayerBaseConfig;

export interface SoftmaxLayerBaseConfig extends LayerBaseConfig {
  /**
   * Integer, axis along which the softmax normalization is applied.
   * Defaults to `-1` (i.e., the last axis).
   */
  axis?: number;
}
