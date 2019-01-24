/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BaseSerialization} from './types';
import {stringDictToArray} from './utils';


// TODO(soergel): Move the CamelCase versions back out of keras_format
// e.g. to src/common.ts.  Maybe even duplicate *all* of these to be pedantic?
/** @docinline */
export type FanMode = 'fanIn'|'fanOut'|'fanAvg';
export const VALID_FAN_MODE_VALUES = ['fanIn', 'fanOut', 'fanAvg'];

// These constants have a snake vs. camel distinction.
export type FanModeSerialization = 'fan_in'|'fan_out'|'fan_avg';

/** @docinline */
export type Distribution = 'normal'|'uniform';
export const VALID_DISTRIBUTION_VALUES = ['normal', 'uniform'];

export type ZerosSerialization = BaseSerialization<'Zeros', {}>;

export type OnesSerialization = BaseSerialization<'Ones', {}>;

export type ConstantConfig = {
  value: number;
};

export type ConstantSerialization =
    BaseSerialization<'Constant', ConstantConfig>;

export type RandomNormalConfig = {
  mean?: number;
  stddev?: number;
  seed?: number;
};

export type RandomNormalSerialization =
    BaseSerialization<'RandomNormal', RandomNormalConfig>;

export type RandomUniformConfig = {
  minval?: number;
  maxval?: number;
  seed?: number;
};

export type RandomUniformSerialization =
    BaseSerialization<'RandomUniform', RandomUniformConfig>;

export type TruncatedNormalConfig = {
  mean?: number;
  stddev?: number;
  seed?: number;
};

export type TruncatedNormalSerialization =
    BaseSerialization<'TruncatedNormal', TruncatedNormalConfig>;

export type VarianceScalingConfig = {
  scale: number;

  mode: FanModeSerialization;
  distribution: Distribution;
  seed?: number;
};

export type VarianceScalingSerialization =
    BaseSerialization<'VarianceScaling', VarianceScalingConfig>;

export type OrthogonalConfig = {
  seed?: number;
  gain?: number;
};

export type OrthogonalSerialization =
    BaseSerialization<'Orthogonal', OrthogonalConfig>;

export type IdentityConfig = {
  gain?: number;
};

export type IdentitySerialization =
    BaseSerialization<'Identity', IdentityConfig>;

// Update NUM_INITIALIZER_OPTIONS in concert (for testing)
export type InitializerSerialization = ZerosSerialization|OnesSerialization|
    ConstantSerialization|RandomUniformSerialization|RandomNormalSerialization|
    TruncatedNormalSerialization|IdentitySerialization|
    VarianceScalingSerialization|OrthogonalSerialization;

export const NUM_INITIALIZER_OPTIONS = 9;

export type InitializerClassName = InitializerSerialization['class_name'];

// This helps guarantee that the Options class below is complete.
export type InitializerOptionMap = {
  [key in InitializerClassName]: string
};

/**
 * List of all known initializer names, along with a string description.
 *
 * Representing this as a class allows both type-checking using the keys and
 * generating an appropriate options array for use in select fields.
 */
class InitializerOptions implements InitializerOptionMap {
  [key: string]: string;
  // tslint:disable:variable-name
  public readonly Zeros = 'Zeros';
  public readonly Ones = 'Ones';
  public readonly Constant = 'Constant';
  public readonly RandomNormal = 'Random Normal';
  public readonly RandomUniform = 'Random Uniform';
  public readonly TruncatedNormal = 'Truncated Normal';
  public readonly VarianceScaling = 'Variance Scaling';
  public readonly Orthogonal = 'Orthogonal';
  public readonly Identity = 'Identity';
  // tslint:enable:variable-name
}

/**
 * An array of `{value, label}` pairs describing the valid initializers.
 *
 * The `value` is the serializable string constant, and the `label` is a more
 * user-friendly description (e.g. for use in UIs).
 */
export const initializerOptions = stringDictToArray(new InitializerOptions());

/**
 * A string array of valid Optimizer class names.
 *
 * This is guaranteed to match the `OptimizerClassName` union type.
 */
export const initializerClassNames = initializerOptions.map((x) => x.value);
