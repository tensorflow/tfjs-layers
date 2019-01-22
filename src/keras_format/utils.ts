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
 * Render the keys and values from a class with string:string attributes as an
 * array of objects.
 */
export function stringDictToArray(options: {[key: string]: string}):
    Array<{value: string, label: string}> {
  const result: Array<{value: string, label: string}> = [];
  for (const key of Object.keys(options)) {
    result.push({value: key, label: options[key]});
  }
  return result;
}
