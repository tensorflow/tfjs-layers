/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {constraintClassNames, NUM_CONSTRAINT_OPTIONS} from './constraint_config';
import {initializerClassNames, NUM_INITIALIZER_OPTIONS} from './initializer_config';
import {NUM_OPTIMIZER_OPTIONS, optimizerClassNames} from './optimizer_config';
import {NUM_REGULARIZER_OPTIONS, regularizerClassNames} from './regularizer_config';

/**
 * These tests guarantee that the type-level represention of various Keras
 * object categories (via union types of the class names) is in sync with the
 * runtime-available representation of the same class names as a string[].
 *
 * Using Optimizers as an example of the pattern:
 *
 * The type structure of `OptimizerOptions` guarantees that every Optimizer
 * class name is present in `OptimizerOptions`, and hence ultimately in
 * `optimizerClassNames`.
 *
 * We also want the converse guarantee, that there are no *additional*
 * elements of `OptimizerOptions` (i.e. keys that match no actual Optimizer
 * class name).  That is impossible to achieve with typing alone (at least in
 * TypeScript 2.9 and without the help of an additional library such as
 * https://github.com/pelotom/runtypes).
 *
 * However it's easy to check that there are no additional keys--without
 * recapitulating all the keys in the test--simply by counting them.  The
 * number of available `Optimizer` classes is determined by the
 * `OptimizerSerialization` union type, so we colocate the constant
 * `NUM_OPTIMIZER_OPTIONS` there.
 */

describe('Constraint options', () => {
  it('constraint map matches available classes', () => {
    expect(constraintClassNames.length).toEqual(NUM_CONSTRAINT_OPTIONS);
  });
});

describe('Initializer options', () => {
  it('constraint map matches available classes', () => {
    expect(initializerClassNames.length).toEqual(NUM_INITIALIZER_OPTIONS);
  });
});

describe('Optimizer options', () => {
  it('options map matches available classes', () => {
    expect(optimizerClassNames.length).toEqual(NUM_OPTIMIZER_OPTIONS);
  });
});

describe('Regularizer options', () => {
  it('options map matches available classes', () => {
    expect(regularizerClassNames.length).toEqual(NUM_REGULARIZER_OPTIONS);
  });
});
