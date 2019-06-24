/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {dispose, Tensor, Tensor1D, tensor1d, tidy} from '@tensorflow/tfjs-core';

export type ClassWeight = {[classIndex: number]: number};
export type ClassWeightMap = {[outputName: string]: ClassWeight};

function standardizeSampleOrClassWeights(
    xWeight: ClassWeight|ClassWeight[]|ClassWeightMap,
    outputNames: string[],
    weightType: 'sampleWeight'|'classWeight'): ClassWeight[] {
  const numOutputs = outputNames.length;
  if (xWeight == null || (Array.isArray(xWeight) && xWeight.length === 0)) {
    return outputNames.map(name => null);
  }
  if (numOutputs === 1) {
    if (Array.isArray(xWeight) && xWeight.length === 1) {
      return xWeight as ClassWeight[];
    } else if (typeof xWeight === 'object' && outputNames[0] in xWeight) {
      return [(xWeight as ClassWeightMap)[outputNames[0]]];
    } else {
      return [xWeight as ClassWeight];
    }
  }
  if (Array.isArray(xWeight)) {
    if (xWeight.length !== numOutputs) {
      throw new Error(
          `Provided ${weightType} is an array of ${xWeight.length} ` +
          `element(s), but the model has ${numOutputs} outputs. ` +
          `Make sure a set of weights is provided for each model output.`);
    }
    return xWeight as ClassWeight[];
  } else if (typeof xWeight === 'object' && Object.keys(xWeight).length > 0 &&
             typeof (xWeight as ClassWeightMap)[Object.keys(xWeight)[0]]
                 === 'object') {
    const output: ClassWeight[] = [];
    outputNames.forEach(outputName => {
      if (outputName in xWeight) {
        output.push((xWeight as ClassWeightMap)[outputName]);
      } else {
        output.push(null) ;
      }
    });
    return output;
  } else {
    throw new Error(
        `The model has multiple (${numOutputs}) outputs, ` +
        `so ${weightType} must be either an array with ` +
        `${numOutputs} elements or an object with ${outputNames} keys. ` +
        `Provided ${weightType} not understood: ${JSON.stringify(xWeight)}`);
  }
}

export function standardizeClassWeights(
    classWeight: ClassWeight|ClassWeight[]|ClassWeightMap,
    outputNames: string[]): ClassWeight[] {
  return standardizeSampleOrClassWeights(
      classWeight, outputNames, 'classWeight');
}

export function standardizeSampleWeights(
    classWeight: ClassWeight|ClassWeight[]|ClassWeightMap,
    outputNames: string[]): ClassWeight[] {
  return standardizeSampleOrClassWeights(
      classWeight, outputNames, 'sampleWeight');
}

// TODO(cais): Doc string.
export async function standardizeWeights(
    y: Tensor,
    sampleWeight?: Tensor,
    classWeight?: ClassWeight,
    sampleWeightMode?: 'temporal'): Promise<Tensor> {
  if (sampleWeight != null || sampleWeightMode != null) {
    throw new Error('Support sampleWeight is not implemented yet');
  }

  if (classWeight != null) {
    // Apply class weights per sample.
    const yClasses: Tensor1D = tidy(() => {
      if (y.shape.length === 1) {
        // Assume class indices.
        return y.clone() as Tensor1D;
      } else if (y.shape.length === 2) {
        if (y.shape[1] > 1) {
          // Assume one-hot encoding of classes.
          const axis = 1;
          return y.argMax(axis);
        } else if (y.shape[1] === 1) {
          // Class indiex.
          return y.reshape([y.shape[0]]);
        } else {
          throw new Error(
              `Encountered unexpected last-dimension size (${y.shape[1]}) ` +
              `during handling of class weights. The size is expected to be ` +
              `>= 1.`);
        }
      } else {
        throw new Error(
            `Unexpected rank of target (y) tensor (${y.rank}) during ` +
            `handling of class weights. The rank is expected to be 1 or 2.`);
      }
    });

    const yClassIndices = Array.from(await yClasses.data());
    dispose(yClasses);
    const classSampleWeight: number[] = [];
    yClassIndices.forEach(classIndex => {
      if (classWeight[classIndex] == null) {
        throw new Error(
            `classWeight must contain all classes in the training data. ` +
            `The class ${classIndex} exists in the data but not in ` +
            `classWeight`);
      } else {
        classSampleWeight.push(classWeight[classIndex]);
      }
    });

    return tensor1d(classSampleWeight, 'float32');
  } else {
    return null;
  }
}
