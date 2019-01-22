/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {stringDictToArray} from './utils';

/**
 * List of all known activation names, along with a string description.
 *
 * Representing this as a class allows both type-checking using the keys and
 * generating an appropriate options array for use in select fields.
 */
export class ActivationOptions {
  [key: string]: string;
  public readonly elu = 'Elu';
  public readonly hardSigmoid = 'Hard Sigmoid';
  public readonly linear = 'Linear';
  public readonly relu = 'Relu';
  public readonly relu6 = 'Relu6';
  public readonly selu = 'Selu';
  public readonly sigmoid = 'Sigmoid';
  public readonly softmax = 'Softmax';
  public readonly softplus = 'Softplus';
  public readonly softsign = 'Softsign';
  public readonly tanh = 'tanh';
}

/**
 * An array of `{value, label}` pairs describing the valid activations.
 *
 * The `value` is the serializable string constant, and the `label` is a more
 * user-friendly description (e.g. for use in UIs).
 */
export const activationOptions = stringDictToArray(new ActivationOptions());

/**
 * A type representing the strings that are valid loss names.
 */
export type ActivationIdentifier = keyof ActivationOptions;
