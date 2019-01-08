/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ConstraintConfig, ConstraintIdentifier} from './constraint_config';
import {InitializerConfig, InitializerIdentifier} from './initializer_config';
import {RegularizerConfig, RegularizerIdentifier} from './regularizer_config';
import {LayerConfig, LayerPrimitiveArgs} from './topology_config';

export interface ReLULayerPrimitiveArgs extends LayerPrimitiveArgs {
  /**
   * Float, the maximum output value.
   */
  maxValue?: number;
}

export type ReLULayerConfig = ReLULayerPrimitiveArgs;

export interface LeakyReLULayerPrimitiveArgs extends LayerPrimitiveArgs {
  /**
   * Float `>= 0`. Negative slope coefficient. Defaults to `0.3`.
   */
  alpha?: number;
}

export type LeakyReLULayerConfig = LeakyReLULayerPrimitiveArgs;

export interface PReLULayerPrimitiveArgs extends LayerPrimitiveArgs {
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

export interface ELULayerPrimitiveArgs extends LayerPrimitiveArgs {
  /**
   * Float `>= 0`. Negative slope coefficient. Defaults to `1.0`.
   */
  alpha?: number;
}

export type ELULayerConfig = ELULayerPrimitiveArgs;

export interface ThresholdedReLULayerPrimitiveArgs extends LayerPrimitiveArgs {
  /**
   * Float >= 0. Threshold location of activation.
   */
  theta?: number;
}

export type ThresholdedReLULayerConfig = ThresholdedReLULayerPrimitiveArgs;

export interface SoftmaxLayerPrimitiveArgs extends LayerPrimitiveArgs {
  /**
   * Integer, axis along which the softmax normalization is applied.
   * Defaults to `-1` (i.e., the last axis).
   */
  axis?: number;
}
