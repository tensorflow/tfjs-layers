/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

function standardizeSampleOrClassWeights(
    xWeight: number[]|number[][]|{[outputName: string]: number[]},
    outputNames: string[],
    weightType: 'sampleWeight'|'classWeight'): number[][] {
  const numOutputs = outputNames.length;
  if (xWeight == null || (Array.isArray(xWeight) && xWeight.length === 0)) {
    return outputNames.map(name => null);
  }
  if (numOutputs === 1) {
    if (Array.isArray(xWeight) && xWeight.length === 1 &&
        Array.isArray(xWeight[0])) {
      return xWeight as number[][];
    } else if (typeof xWeight === 'object' && outputNames[0] in xWeight) {
      return [xWeight[outputNames[0]]];
    } else {
      return [xWeight as number[]];
    }
  }
  if (Array.isArray(xWeight)) {
    if (xWeight.length != numOutputs) {
      throw new Error(
          `Provided ${weightType} is an array of ${xWeight.length} ` +
          `element(s), but the model has ${numOutputs} outputs. ` +
          `Make sure a set of weights is provided for each model output.`);
    }
    // TODO(cais): Check unique length in `xWeight`?
    return xWeight as number[][];
  } else if (typeof xWeight === 'object') {
    return outputNames.map(outputName => xWeight[outputName]);
  } else {
    throw new Error(
        `The model has multiple ${numOutputs} outputs, ` +
        `so ${weightType} must be either an array with ` +
        `${numOutputs} elements or an object with ${outputNames} keys. ` +
        `Provided ${weightType} not understood: ${JSON.stringify(xWeight)}`);
  }
}

export function standardizeClassWeights(
    classWeight: number[]|number[][]|{[outputName: string]: number[]},
    output_names: string[]): number[][] {
  return standardizeSampleOrClassWeights(
      classWeight, output_names, 'classWeight');
}

export function standardizeSampleWeights(
    classWeight: number[]|number[][]|{[outputName: string]: number[]},
    output_names: string[]): number[][] {
  return standardizeSampleOrClassWeights(
      classWeight, output_names, 'sampleWeight');
}
