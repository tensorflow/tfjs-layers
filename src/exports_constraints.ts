/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
// tslint:disable-next-line:max-line-lengthMinMaxNormArgs
import {Constraint, MaxNorm, MaxNormArgs, MinMaxNorm, MinMaxNormArgs, NonNeg, UnitNorm, UnitNormArgs} from './constraints';

/**
 * @doc {
 *   heading: 'Constraints',
 *   namespace: 'constraints',
 *   useDocsFrom: 'MaxNorm',
 *   configParamIndices: [0]
 * }
 */
export function maxNorm(args: MaxNormArgs): Constraint {
  return new MaxNorm(args);
}

/**
 * @doc {
 *   heading: 'Constraints',
 *   namespace: 'constraints',
 *   useDocsFrom: 'UnitNorm',
 *   configParamIndices: [0]
 * }
 */
export function unitNorm(args: UnitNormArgs): Constraint {
  return new UnitNorm(args);
}

/**
 * @doc {
 *   heading: 'Constraints',
 *   namespace: 'constraints',
 *   useDocsFrom: 'NonNeg'
 * }
 */
export function nonNeg(): Constraint {
  return new NonNeg();
}

/**
 * @doc {
 *   heading: 'Constraints',
 *   namespace: 'constraints',
 *   useDocsFrom: 'MinMaxNormConfig',
 *   configParamIndices: [0]
 * }
 */
export function minMaxNorm(config: MinMaxNormArgs): Constraint {
  return new MinMaxNorm(config);
}
