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

export interface ConstantBaseConfig {
  /** The value for each element in the variable. */
  value: number;
}

export type ConstantConfig = ConstantBaseConfig;

export interface RandomUniformBaseConfig {
  /** Lower bound of the range of random values to generate. */
  minval?: number;
  /** Upper bound of the range of random values to generate. */
  maxval?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

export type RandomUniformConfig = RandomUniformBaseConfig;

export interface RandomNormalBaseConfig {
  /** Mean of the random values to generate. */
  mean?: number;
  /** Standard deviation of the random values to generate. */
  stddev?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

export type RandomNormalConfig = RandomNormalBaseConfig;

export interface TruncatedNormalBaseConfig {
  /** Mean of the random values to generate. */
  mean?: number;
  /** Standard deviation of the random values to generate. */
  stddev?: number;
  /** Used to seed the random generator. */
  seed?: number;
}

export type TruncatedNormalConfig = TruncatedNormalBaseConfig;

export interface IdentityBaseConfig {
  /**
   * Multiplicative factor to apply to the identity matrix.
   */
  gain?: number;
}

export type IdentityConfig = IdentityBaseConfig;

export interface VarianceScalingBaseConfig {
  /** Scaling factor (positive float). */
  scale: number;

  /** Fanning mode for inputs and outputs. */
  mode: FanMode;

  /** Probabilistic distribution of the values. */
  distribution: Distribution;

  /** Random number generator seed. */
  seed?: number;
}

export type VarianceScalingConfig = VarianceScalingBaseConfig;

export interface SeedOnlyInitializerBaseConfig {
  /** Random number generator seed. */
  seed?: number;
}

export type SeedOnlyInitializerConfig = SeedOnlyInitializerBaseConfig;

export interface OrthogonalBaseConfig extends SeedOnlyInitializerBaseConfig {
  /**
   * Multiplicative factor to apply to the orthogonal matrix. Defaults to 1.
   */
  gain?: number;
}

export type OrthogonalConfig = OrthogonalBaseConfig;

export type InitializerConfig = ConstantConfig|RandomUniformConfig|
    RandomNormalConfig|TruncatedNormalConfig|IdentityConfig|
    VarianceScalingConfig|SeedOnlyInitializerConfig|OrthogonalConfig;
