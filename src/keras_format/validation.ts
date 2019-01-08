/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Helper function used to build type union/enum run-time checkers.
 * @param values The list of allowed values.
 * @param label A string name for the type
 * @param value The value to test.
 * @throws ValueError: If the value is not in values nor `undefined`/`null`.
 */
export function checkStringTypeUnionValue(
    values: string[], label: string, value: string): void {
  if (value == null) {
    return;
  }
  if (values.indexOf(value) < 0) {
    throw new Error(`${value} is not a valid ${label}.  Valid values are ${
        values} or null/undefined.`);
  }
}
