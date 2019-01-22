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

export type L1L2Config = {
  l1?: number;
  l2?: number;
};

export type L1L2Serialization = BaseSerialization<'L1L2', L1L2Config>;

export type L1Config = {
  l1?: number;
};

export type L1Serialization = BaseSerialization<'L1', L1Config>;

export type L2Config = {
  l2?: number;
};

export type L2Serialization = BaseSerialization<'L2', L2Config>;

export type RegularizerSerialization =
    L1L2Serialization|L1Serialization|L2Serialization;

export type RegularizerClassName = RegularizerSerialization['class_name'];

/**
 * List of all known regularizer names, along with a string description.
 *
 * Representing this as a class allows both type-checking using the keys and
 * generating an appropriate options array for use in select fields.
 */
export class RegularizerOptions {
  [key: string]: string;
  // tslint:disable:variable-name
  public readonly L1L2 = 'L1L2';
  public readonly L1 = 'L1';
  public readonly L2 = 'L2';
  // tslint:enable:variable-name
}

/**
 * An array of `{value, label}` pairs describing the valid regularizers.
 *
 * The `value` is the serializable string constant, and the `label` is a more
 * user-friendly description (e.g. for use in UIs).
 */
export const regularizerOptions = stringDictToArray(new RegularizerOptions());

/**
 * A type representing the strings that are valid regularizer names.
 */
// TODO(soergel): test assert this is identical to RegularizerClassName
// export type RegularizerIdentifier = keyof RegularizerOptions;
