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
import {oneHot, serialization, Tensor, Tensor1D, tidy,} from '@tensorflow/tfjs-core';
import {Layer, LayerConfig} from '../engine/topology';
import {Kwargs, Shape} from '../types';
import * as generic_utils from '../utils/generic_utils';
// import {StringTensor} from './string_tensor';

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


export interface VocabularyLayerConfig extends LayerConfig {
  hashVocabSize: number;
  knownVocabSize: number;
}

/*
// A vocabulary layer is a map from strings to integers in the range
// [0, buckets), where buckets is hashVocabSize + knownVocabSize.
export class VocabularyLayer extends Layer {
  static className = 'VocabularyLayer';
  readonly hashVocabularySize = 0;
  readonly knownVocabSize: number;

  // Map of words in the known vocabulary.  Key is words in the known
  // vocabulary.  Value is a counter object used during fitting of the
  // vocabulary to a datset.
  private knownVocabulary: Map<string, number>;

  constructor(config: VocabularyLayerConfig) {
    super(config);
  }

  public build(inputShape: Shape|Shape[]): void {
    this.built = true;
  }

  strToIntFn(key: string): number {
    return this.knownVocabulary.get(key);
  }

  // DOING DOING DOING...
  // Expand the definition of what a 'Layer' can do to include working with
  // StringTensors...
  // DOING DOING DOING...


  call(inputs: StringTensor|StringTensor[], kwargs: Kwargs): Tensor|Tensor[] {
    const stringTensor = generic_utils.getExactlyOneTensor(inputs);
    return tidy(() => {
      const stringValues = stringTensor.dataSync();
      const intValues = stringValues.map(this.strToIntFn);

      // Index each word into the known Vocabulary.

      // If more than one hashVocabularySize, for each word that was *not* found
      // in the known vocabulary, hash the word into hashVocabularySize into
      // hashVocabularySize buckets and return that.

      return tensor1d([1]);
      return tensor(intValues, stringTensor.shape, 'int32'
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
serialization.SerializationMap.register(VocabularyLayer);
*/
