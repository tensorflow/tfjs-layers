/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ActivationIdentifier} from '../activations';

import {ConstraintConfig, ConstraintIdentifier} from './constraint_config';
import {InitializerConfig, InitializerIdentifier} from './initializer_config';
import {RegularizerConfig, RegularizerIdentifier} from './regularizer_config';
import {LayerPrimitiveArgs} from './topology_config';
import {Shape} from './types';

export interface DropoutLayerPrimitiveArgs extends LayerPrimitiveArgs {
  /** Float between 0 and 1. Fraction of the input units to drop. */
  rate: number;

  /**
   * Integer array representing the shape of the binary dropout mask that will
   * be multiplied with the input.
   *
   * For instance, if your inputs have shape `(batchSize, timesteps, features)`
   * and you want the dropout mask to be the same for all timesteps, you can use
   * `noise_shape=(batch_size, 1, features)`.
   */
  noiseShape?: number[];

  /** An integer to use as random seed. */
  seed?: number;
}

export type DropoutLayerConfig = DropoutLayerPrimitiveArgs;

export interface DenseLayerPrimitiveArgs extends LayerPrimitiveArgs {
  /** Positive integer, dimensionality of the output space. */
  units: number;

  /**
   * Activation function to use.
   *
   * If unspecified, no activation is applied.
   */
  activation?: ActivationIdentifier;

  /** Whether to apply a bias. */
  useBias?: boolean;

  /**
   * If specified, defines inputShape as `[inputDim]`.
   */
  inputDim?: number;
}

export interface DenseLayerConfig extends DenseLayerPrimitiveArgs {
  /**
   * If specified, defines inputShape as `[inputDim]`.
   */
  inputDim?: number;

  /**
   * Initializer for the dense kernel weights matrix.
   */
  kernelInitializer?: InitializerIdentifier|InitializerConfig;
  /**
   * Initializer for the bias vector.
   */
  biasInitializer?: InitializerIdentifier|InitializerConfig;

  /**
   * Constraint for the kernel weights.
   */
  kernelConstraint?: ConstraintIdentifier|ConstraintConfig;

  /**
   * Constraint for the bias vector.
   */
  biasConstraint?: ConstraintIdentifier|ConstraintConfig;

  /**
   * Regularizer function applied to the dense kernel weights matrix.
   */
  kernelRegularizer?: RegularizerIdentifier|RegularizerConfig;

  /**
   * Regularizer function applied to the bias vector.
   */
  biasRegularizer?: RegularizerIdentifier|RegularizerConfig;

  /**
   * Regularizer function applied to the activation.
   */
  activityRegularizer?: RegularizerIdentifier|RegularizerConfig;
}


export interface ActivationLayerPrimitiveArgs extends LayerPrimitiveArgs {
  /**
   * Name of the activation function to use.
   */
  activation: ActivationIdentifier;
}

export interface ActivationLayerConfig extends ActivationLayerPrimitiveArgs {}

export interface RepeatVectorLayerPrimitiveArgs extends LayerPrimitiveArgs {
  /**
   * The integer number of times to repeat the input.
   */
  n: number;
}

export interface RepeatVectorLayerConfig extends
    RepeatVectorLayerPrimitiveArgs {}

export interface ReshapeLayerPrimitiveArgs extends LayerPrimitiveArgs {
  /** The target shape. Does not include the batch axis. */
  targetShape: Shape;
}

export interface ReshapeLayerConfig extends ReshapeLayerPrimitiveArgs {}

export interface PermuteLayerPrimitiveArgs extends LayerPrimitiveArgs {
  /**
   * Array of integers. Permutation pattern. Does not include the
   * sample (batch) dimension. Index starts at 1.
   * For instance, `[2, 1]` permutes the first and second dimensions
   * of the input.
   */
  dims: number[];
}
