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

export type MaxNormConfig = {
  max_value?: number;
  axis?: number;
};

export type MaxNormSerialization = BaseSerialization<'MaxNorm', MaxNormConfig>;

export type UnitNormConfig = {
  axis?: number;
};

export type UnitNormSerialization =
    BaseSerialization<'UnitNorm', UnitNormConfig>;

export type NonNegSerialization = BaseSerialization<'NonNeg', null>;

export type MinMaxNormConfig = {
  min_value?: number;
  max_value?: number;
  axis?: number;
  rate?: number;
};

export type MinMaxNormSerialization =
    BaseSerialization<'MinMaxNorm', MinMaxNormConfig>;

// Update NUM_CONSTRAINT_OPTIONS in concert (for testing)
export type ConstraintSerialization = MaxNormSerialization|NonNegSerialization|
    UnitNormSerialization|MinMaxNormSerialization;

export const NUM_CONSTRAINT_OPTIONS = 4;

export type ConstraintClassName = ConstraintSerialization['class_name'];

// This helps guarantee that the Options class below is complete.
export type ConstraintOptionMap = {
  [key in ConstraintClassName]: string
};

/**
 * List of all known constraint names, along with a string description.
 *
 * Representing this as a class allows both type-checking using the keys and
 * automatically translating to human readable display names where needed.
 */
class ConstraintOptions implements ConstraintOptionMap {
  [key: string]: string;
  // tslint:disable:variable-name
  public readonly MaxNorm = 'Max Norm';
  public readonly UnitNorm = 'Unit Norm';
  public readonly NonNeg = 'Non-negative';
  public readonly MinMaxNorm = 'Min-Max Norm';
  // tslint:enable:variable-name
}

/**
 * An array of `{value, label}` pairs describing the valid constraints.
 *
 * The `value` is the serializable string constant, and the `label` is a more
 * user-friendly description (e.g. for use in UIs).
 */
export const constraintOptions = stringDictToArray(new ConstraintOptions());

/**
 * A string array of valid Optimizer class names.
 *
 * This is guaranteed to match the `OptimizerClassName` union type.
 */
export const constraintClassNames = constraintOptions.map((x) => x.value);
