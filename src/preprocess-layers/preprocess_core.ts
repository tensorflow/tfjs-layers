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
 * TensorFlow.js Layers: Basic Layers.
 */

// tslint:disable:max-line-length
import {oneHot, serialization, Tensor, tensor, Tensor1D, tidy,} from '@tensorflow/tfjs-core';

import {Layer, LayerConfig} from '../engine/topology';
import {ValueError} from '../errors';
import {Kwargs, Shape} from '../types';
import * as generic_utils from '../utils/generic_utils';

import {StringTensor} from './string_tensor';

export interface OneHotLayerConfig extends LayerConfig {
  /** Positive integer, dimensionality of the output space. */
  units: number;
}


/**
 * Requires input of shape [batch].  Produces output of shape [batch, units]
 */
export class OneHot extends Layer {
  static className = 'OneHot';
  readonly units: number;

  constructor(config: OneHotLayerConfig) {
    super(config);
    this.units = config.units;

    this.inputSpec = [{minNDim: 1}];
  }

  public build(inputShape: Shape|Shape[]): void {
    this.inputSpec = [{minNDim: 1}];
    this.built = true;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = generic_utils.getExactlyOneShape(inputShape) as Shape;
    const outputShape = [inputShape[0], this.units];
    return outputShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);

      const input = generic_utils.getExactlyOneTensor(inputs);
      const output = oneHot(input as Tensor1D, this.units);
      // TODO(bileschi) remove type fix once oneHot is consistent.
      // https://github.com/tensorflow/tfjs/issues/435
      return output.asType('float32');
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      units: this.units,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(OneHot);


export interface VocabLayerConfig extends LayerConfig {
  hashVocabSize?: number;
  knownVocabSize: number;
}

// TODO(bileschi): Replace with the hash op used in c++ / py tensorflow here:
// core/lib/hash/hash.h
export function vocabHash64(s: string) {
  let hash = 0xDECAFCAFFE, i, chr;
  if (s.length === 0) return hash;
  for (i = 0; i < s.length; i++) {
    chr = s.charCodeAt(i);
    hash = ((hash << 5) - hash) + chr;
    hash |= 0;  // Convert to 32bit integer
  }
  return Math.abs(hash);
}


// A `VocabLayer` is a fittable map from strings to integers in the range
// [0, buckets), where buckets is hashVocabSize + knownVocabSize.
export class VocabLayer extends Layer {
  static className = 'VocabularyLayer';
  readonly hashVocabSize: number;
  readonly knownVocabSize: number;

  // Map of words in the known vocabulary.  Key is words in the known
  // vocabulary.  Value is a counter object used during fitting of the
  // vocabulary to a datset.
  private knownVocab: Map<string, number>;

  constructor(config: VocabLayerConfig) {
    super(config);
    this.knownVocabSize = config.knownVocabSize;
    this.hashVocabSize = config.hashVocabSize | 0;
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    if (this.knownVocab == null) {
      // TODO(bileschi): knownVocab initialization should go here.
      this.knownVocab = new Map<string, number>();
    }
    this.built = true;
  }

  strToIntFn(key: string): number {
    // Index each word into the known Vocab.

    // If hashVocabSize is greater than one, for each word that was *not*
    // found in the known vocabulary, hash the word into hashVocabSize
    // buckets and return that.
    if (this.knownVocab.has(key)) {
      return this.knownVocab.get(key);
    } else {
      if (this.hashVocabSize <= 0) {
        throw new ValueError('Key not in vocab.  Configure hashVocabSize > 0.');
      }
      if (this.hashVocabSize === 1) {
        // Out-of-vocabulary buckets begin after known vocabulary buckets.
        return this.knownVocabSize;
      }
      // hashVocabSize > 1;
      // Out-of-vocabulary buckets begin after known vocabulary buckets.
      return this.hashBucketFn(key, this.hashVocabSize) + this.knownVocabSize;
    }
  }

  // TODO(bileschi): Clone hash functions from tensorflow string ops.
  // .../tensorflow/python/ops/lookup_ops.py#L841
  hashBucketFn(s: string, numBuckets: number) {
    return vocabHash64(s) % numBuckets;
  }

  call(inputs: StringTensor|StringTensor[], kwargs: Kwargs): Tensor|Tensor[] {
    let stringTensor: StringTensor;
    if (Array.isArray(inputs)) {
      if (inputs.length !== 1) {
        throw new ValueError(
            `Vocab initializer expected Tensor length to be 1; got ${
                inputs.length}`);
      }
      stringTensor = inputs[0];
    } else {
      stringTensor = inputs as StringTensor;
    }
    return tidy(() => {
      const intValues: number[] = [];
      stringTensor.stringValues.forEach(s => {
        intValues.push(this.strToIntFn(s));
      });
      return tensor(intValues, stringTensor.shape, 'int32');
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      hashVocabSize: this.hashVocabSize,
      knownVocabSize: this.knownVocabSize,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  // TODO(bileschi):  I added this for testing.  Do we want something like this?
  public setVocab(newVocab: Map<string, number>) {
    this.knownVocab = newVocab;
  }

  // TODO(bileschi):  I added this for testing.  Do we want something like this?
  public getVocab(): Map<string, number> {
    return this.knownVocab;
  }
}
serialization.SerializationMap.register(VocabLayer);
