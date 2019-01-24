/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// https://stackoverflow.com/questions/52085454/typescript-define-a-union-type-from-an-array-of-strings/52085658
export function stringLiteralArray<T extends string>(a: T[]) {
  return a;
}
