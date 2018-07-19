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
import {ConfigDict, Serializable} from '@tensorflow/tfjs-core/dist/serialization';

import {Layer, LayerConfig} from '../engine/topology';
import {ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier} from '../initializers';
import {Kwargs, Shape} from '../types';
import * as type_utils from '../utils/types_utils';

import {StringTensor} from './string_tensor';

export interface OneHotLayerConfig extends LayerConfig {
  /** Positive integer, dimensionality of the output space. */
  units: number;
}

/**
 * Preprocessing layers are distinct from Layers in that they are not
 * affected by back-propagation.  They are not affected by .fit.  They may be
 * affected by `Model.fitUnsupervised`.
 */
export abstract class PreprocessingLayer extends Layer {}

/**
 * Requires input of shape [batch].  Produces output of shape [batch, units]
 */
export class OneHot extends PreprocessingLayer {
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
    inputShape = type_utils.getExactlyOneShape(inputShape) as Shape;
    const outputShape = [inputShape[0], this.units];
    return outputShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);

      const input = type_utils.getExactlyOneTensor(inputs);
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

export class VocabLayerOptimizer extends Serializable {
  static className = 'VocabLayerOptimizer';
  public wordCount: Map<string, number>;

  constructor() {
    super();
    this.wordCount = new Map<string, number>();
  }

  getConfig(): ConfigDict {
    return {};
  }

  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict): T {
    return new cls();
  }


  public updateCounts(words: StringTensor) {
    for (const word of words.stringValues) {
      // if string is a key in wordCount, update it.
      if (this.wordCount.has(word)) {
        this.wordCount.set(word, this.wordCount.get(word) + 1);
      } else {
        this.wordCount.set(word, 1);
      }
    }
  }

  // Sort by greater count first, alphabetically second.
  protected _compareCountWords(a: [number, string], b: [number, string]):
      number {
    if (a[0] === b[0]) {
      // If the counts are the same, a should come first if it's alphabetically
      // first.
      return +(a[1] > b[1]);
    } else {
      // Otherwise a is should come first if its count is larger.
      return +(a[0] < b[0]);
    }
  }

  // Modifies provided vocab to replace low-count words with higher-count words.
  public updateVocab(vocab: Map<string, number>, knownVocabSize: number) {
    // TODO(bileschi): There is room for optimization here by, e.g., only adding
    // and removing values from the map and never moving them.
    vocab.clear();
    // 1. Convert this.wordCount into (count, word) pairs.
    const countWordPairs: Array<[number, string]> = [];
    this.wordCount.forEach((valUnused, key) => {
      countWordPairs.push([this.wordCount.get(key), key]);
    });
    // 2. sort countWordPairs by descending count.
    countWordPairs.sort(this._compareCountWords);
    // 3. Insert the top knownVocabSize words into vocab
    let numInserted = 0;
    for (const countAndWord of countWordPairs) {
      vocab.set(countAndWord[1], numInserted);
      numInserted++;
      if (numInserted >= knownVocabSize) {
        break;
      }
    }
  }
}
serialization.SerializationMap.register(VocabLayerOptimizer);

export interface VocabLayerConfig extends LayerConfig {
  hashVocabSize?: number;
  knownVocabSize: number;
  vocabInitializer?: InitializerIdentifier|Initializer;
  optimizer?: VocabLayerOptimizer;
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
export class VocabLayer extends PreprocessingLayer {
  static className = 'VocabularyLayer';
  readonly hashVocabSize: number;
  readonly knownVocabSize: number;
  private vocabInitializer: Initializer;
  private optimizer: VocabLayerOptimizer;

  readonly DEFAULT_VOCAB_INITIALIZER: InitializerIdentifier = 'rainbowVocab';


  // Map of words in the known vocabulary.  Key is words in the known
  // vocabulary.  Value is the intenger associated with that word.
  private knownVocab: Map<string, number>;

  constructor(config: VocabLayerConfig) {
    super(config);
    this.dtype = 'string';
    this.knownVocabSize = config.knownVocabSize;
    this.hashVocabSize = config.hashVocabSize | 0;
    this.vocabInitializer = getInitializer(
        config.vocabInitializer || this.DEFAULT_VOCAB_INITIALIZER);
    // Like Model, optimzer may be undefined here if it was not provided via
    // config.
    this.optimizer = config.optimizer;
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = type_utils.getExactlyOneShape(inputShape);
    // console.log('VocabLayer.build');
    if (this.knownVocab == null && this.knownVocabSize &&
        this.knownVocabSize > 0) {
      // console.log('VocabLayer.build -- apply vocabInitializer');
      const vocabTensor = this.vocabInitializer.apply(
                              [this.knownVocabSize], 'string') as StringTensor;
      // console.log('VocabLayer initialization complete');
      this.knownVocab = new Map<string, number>();
      for (let i = 0; i < vocabTensor.size; i++) {
        this.knownVocab.set(vocabTensor.get(i), i);
      }
    }
    this.built = true;
  }

  // TODO(bileschi): This should probably return a `History` or some other way
  // of keeping track of what happens
  public fitUnsupervised(x: StringTensor): void {
    if (this.optimizer) {
      if (!(this.built)) {
        this.build(x.shape);
      }
      this.optimizer.updateCounts(x);
      this.optimizer.updateVocab(this.knownVocab, this.knownVocabSize);
    } else {
      throw new ValueError(
          '.fit() called on VocabLayer with no optimizer.' +
          '  VocabLayer must be configured with an optimizer to be fittable');
    }
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
            `Vocab layer expected one tensor input; got ${inputs.length}`);
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
