/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {checkStringTypeUnionValue} from './validation';

/** @docinline */
export type InitializerIdentifier = 'constant'|'glorotNormal'|'glorotUniform'|
    'heNormal'|'identity'|'leCunNormal'|'ones'|'orthogonal'|'randomNormal'|
    'randomUniform'|'truncatedNormal'|'varianceScaling'|'zeros'|string;

/** @docinline */
export type FanMode = 'fanIn'|'fanOut'|'fanAvg';
export const VALID_FAN_MODE_VALUES = ['fanIn', 'fanOut', 'fanAvg'];
export function checkFanMode(value?: string): void {
  checkStringTypeUnionValue(VALID_FAN_MODE_VALUES, 'FanMode', value);
}

/** @docinline */
export type Distribution = 'normal'|'uniform';
export const VALID_DISTRIBUTION_VALUES = ['normal', 'uniform'];
export function checkDistribution(value?: string): void {
  checkStringTypeUnionValue(VALID_DISTRIBUTION_VALUES, 'Distribution', value);
}

export interface ConstantPrimitiveArgs {
  /** The value for each element in the variable. */
  value: number;
}

export type ConstantConfig = ConstantPrimitiveArgs;

export interface RandomUniformPrimitiveArgs {
  /** Lower bound of the range of random values to generate. */
  minval?: number;
  /** Upper bound of the range of random values to generate. */
  maxval?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

export type RandomUniformConfig = RandomUniformPrimitiveArgs;

export interface RandomNormalPrimitiveArgs {
  /** Mean of the random values to generate. */
  mean?: number;
  /** Standard deviation of the random values to generate. */
  stddev?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

export type RandomNormalConfig = RandomNormalPrimitiveArgs;

export interface TruncatedNormalPrimitiveArgs {
  /** Mean of the random values to generate. */
  mean?: number;
  /** Standard deviation of the random values to generate. */
  stddev?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

export type TruncatedNormalConfig = TruncatedNormalPrimitiveArgs;

export interface IdentityPrimitiveArgs {
  /**
   * Multiplicative factor to apply to the identity matrix.
   */
  gain?: number;
}

export type IdentityConfig = IdentityPrimitiveArgs;

export interface VarianceScalingPrimitiveArgs {
  /** Scaling factor (positive float). */
  scale: number;

  /** Fanning mode for inputs and outputs. */
  mode: FanMode;

  /** Probabilistic distribution of the values. */
  distribution: Distribution;

  /** Random number generator seed. */
  seed?: number;
}

export type VarianceScalingConfig = VarianceScalingPrimitiveArgs;

export interface SeedOnlyInitializerPrimitiveArgs {
  /** Random number generator seed. */
  seed?: number;
}

export type SeedOnlyInitializerConfig = SeedOnlyInitializerPrimitiveArgs;

export interface OrthogonalPrimitiveArgs extends
    SeedOnlyInitializerPrimitiveArgs {
  /**
   * Multiplicative factor to apply to the orthogonal matrix. Defaults to 1.
   */
  gain?: number;
}

export type OrthogonalConfig = OrthogonalPrimitiveArgs;

export type InitializerConfig = ConstantConfig|RandomUniformConfig|
    RandomNormalConfig|TruncatedNormalConfig|IdentityConfig|
    VarianceScalingConfig|SeedOnlyInitializerConfig|OrthogonalConfig;
