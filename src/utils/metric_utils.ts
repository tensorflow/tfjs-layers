/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {LossOrMetricFn} from '../types';
import * as util from './generic_utils';
import {lossesMap} from '../losses';
import {metricsMap} from '../metrics';

/**
 * Get the shortcut function name.
 *
 * If the fn name is a string,
 *   directly return the string name.
 * If the function is included in metricsMap or lossesMap,
 *   return key of the map.
 *   - If the function relative to multiple keys,
 *     return the first found key as the function name.
 *   - If the function exsits in both lossesMap and metricsMap,
 *     search lossesMap first.
 * If the function is not included in metricsMap or lossesMap,
 *   return the function name.
 *
 * @param fn loss function, metric function, or short cut name.
 * @returns Loss or Metric name in string.
 */
export function getLossOrMetricFnName(fn: string|LossOrMetricFn): string {
  util.assert(
      fn !== null,
      `Unknown LossOrMetricFn ${fn}`
  );
  if (typeof fn === 'string') {
    return fn;
  } else {
    let foundFnName = false;
    let fnName = '';
    for (const key of Object.keys(lossesMap)) {
      if (lossesMap[key] === fn) {
        foundFnName = true;
        fnName = key;
        break;
      }
    }
    if (foundFnName) {
      return fnName;
    }
    for (const key of Object.keys(metricsMap)) {
      if (metricsMap[key] === fn) {
        foundFnName = true;
        fnName = key;
        break;
      }
    }
    if (foundFnName) {
      return fnName;
    }
    return (fn as Function).name;
  }
}