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
import {oneHot, Rank, serialization, ShapeMap, Tensor, Tensor1D, tidy, util} from '@tensorflow/tfjs-core';
import {doc} from '@tensorflow/tfjs-core';
import {Layer, LayerConfig} from '../engine/topology';
import {Kwargs, Shape} from '../types';
import * as generic_utils from '../utils/generic_utils';

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

// String type stuff below here;
////////////////////////////////

export type StringArray = string[]|string[][]|string[][][]|string[][][][]|
    string[][][][][]|string[][][][][][];

// TODO:(bileschi): use the core version of this in utils.ts
export function computeStrides(shape: number[]): number[] {
  const rank = shape.length;
  if (rank < 2) {
    return [];
  }

  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  const strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// TODO:(bileschi): use the core version of this in utils.ts
export function arraysEqual(n1: number[], n2: number[]) {
  if (n1.length !== n2.length) {
    return false;
  }
  for (let i = 0; i < n1.length; i++) {
    if (n1[i] !== n2[i]) {
      return false;
    }
  }
  return true;
}

// TODO:(bileschi): use the core version of this in utils.ts
export function assertShapesMatch(
    shapeA: number[], shapeB: number[], errorMessagePrefix = ''): void {
  util.assert(
      arraysEqual(shapeA, shapeB),
      errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
}

// TODO:(bileschi): use the core version of this in utils.ts
export function assertNonNull(a: string|string[]|string[][]): void {
  util.assert(
      a != null,
      `The input to the tensor constructor must be a non-null value.`);
}

// TODO:(bileschi): use the core version of this in utils.ts
export function inferShape(val: string|StringArray): number[] {
  if (!Array.isArray(val)) {
    return [];  // Scalar.
  }
  const shape: number[] = [];
  while (val instanceof Array) {
    shape.push(val.length);
    val = val[0];
  }
  return shape;
}


// This intentionally does not extend `Tensor`, because `Tensor` comes with the
// machinery for operating ops within an envirionment.

export class StringTensor<R extends Rank = Rank> {
  readonly shape: ShapeMap[R];
  readonly size: number;
  /** The rank type for the array (see `Rank` enum). */
  readonly rankType: R;
  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated/\
   * numpy.ndarray.strides.html
   */
  readonly strides: number[];
  readonly stringValues: string[];

  get rank(): number {
    return this.shape.length;
  }

  protected constructor(shape: ShapeMap[R], stringValues?: string[]) {
    this.size = util.sizeFromShape(shape);
    if (stringValues != null) {
      util.assert(
          this.size === stringValues.length,
          `Constructing tensor of shape (${this.size}) should match the ` +
              `length of stringValues (${stringValues.length})`);
      this.stringValues = stringValues;
    }
    this.shape = shape.slice();
    this.strides = computeStrides(shape);
    this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher') as R;
  }

  /**
   * Makes a new tensor with the provided shape and values. Values should be in
   * a flat array.
   */
  static make<T extends StringTensor<R>, R extends Rank = Rank>(
      shape: ShapeMap[R], data: string[]): T {
    return new StringTensor(shape, data) as T;
  }

  /**
   * Returns the value in the `StringTensor` at the provided location.
   *
   * @param locs The location indices.
   */
  get(...locs: number[]) {
    util.assert(
        locs.length === this.rank,
        'Number of coordinates in get() must match the rank of the tensor');
    if (locs.length === 0) {
      locs = [0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.stringValues[index];
  }
}

export type StringTensor2D = StringTensor<Rank.R2>;
export class PreprocessingExports {
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static stringTensor2D(values: string[]|string[][], shape?: [number, number]):
      StringTensor2D {
    assertNonNull(values);
    if (shape != null && shape.length !== 2) {
      throw new Error('stringTensor2d() requires shape to have two numbers');
    }
    const inferredShape = inferShape(values);
    if (inferredShape.length !== 2 && inferredShape.length !== 1) {
      throw new Error(
          'stringTensor2d() requires stringValues to be string[][] or string[]');
    }
    if (inferredShape.length === 1 && shape == null) {
      throw new Error(
          'tensor2d() requires shape to be provided when `values` ' +
          'are a flat/TypedArray');
    }
    shape = shape || inferredShape as [number, number];
    return PreprocessingExports.stringTensor(values, shape);
  }

  /**
   * Creates a `StringTensor` with the provided values, shape and dtype.
   *
   * ```js
   * // Pass an array of values to create a vector.
   * tf.stringTensor(["hello", "world"]).print();
   * ```
   *
   * ```js
   * // Pass a nested array of values to make a matrix or a higher
   * // dimensional tensor.
   * tf.stringTensor([["saturday", "night", ["fe", "ver"]]).print();
   * ```
   *
   * ```js
   * // Pass a flat array and specify a shape yourself.
   * tf.stringTensor(["AA", "AB", "BA", "BB"], [2, 2]).print();
   * ```
   *
   * @param values The values of the tensor. Can be a nested array of strings,
   *     or a flat array.
   * @param shape The shape of the tensor. Optional. If not provided,
   *   it is inferred from `values`.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static stringTensor<R extends Rank>(
      values: string|StringArray, shape?: ShapeMap[R]): StringTensor<R> {
    const inferredShape = inferShape(values);
    if (shape != null && inferredShape.length !== 1) {
      assertShapesMatch(
          shape, inferredShape,
          `Error creating a new Tensor. ` +
              `Inferred shape (${inferredShape}) does not match the ` +
              `provided shape (${shape}). `);
    }
    if (!Array.isArray(values)) {
      values = [values] as string[];
    }
    shape = shape || inferredShape;
    return StringTensor.make(shape, flattenStringArray(values));
  }
}

// TODO:(bileschi): use the core version of this in utils.ts
// tslint:disable-next-line:no-any
export interface RecursiveArray<T extends any> {
  [index: number]: T|RecursiveArray<T>;
}

// TODO:(bileschi): use the core version of this in utils.ts
export function flattenStringArray(
    arr: string|RecursiveArray<string>, ret: string[] = []): string[] {
  if (Array.isArray(arr)) {
    for (let i = 0; i < arr.length; ++i) {
      flattenStringArray(arr[i], ret);
    }
  } else {
    ret.push(arr as string);
  }
  return ret;
}

/*
export interface VocabularyLayerConfig extends LayerConfig {
  hashVocabSize: number;
  knownVocabSize: number;
}

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
