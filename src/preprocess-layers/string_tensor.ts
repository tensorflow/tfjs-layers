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
 * TensorFlow.js StringTensor.
 */

// tslint:disable:max-line-length
import {Rank, ShapeMap, util} from '@tensorflow/tfjs-core';
import {ValueError} from '../errors';

/////////////////////////
//// UTILS AND STUFF ////
/////////////////////////

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
export function assertNonNull(a: string|string[]|string[][]|string[][][]|
                              string[][][][]|string[][][][][]|
                              string[][][][][][]): void {
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

///////////////////////////////////
///// STRING TENSOR DEFINITION ////
///////////////////////////////////

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
  // dtype is required here for duck-type compatibility with Tensor.
  readonly dtype = 'string';

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
    } else {
      // Initialize stringValues to an emtpy array.
      this.stringValues = [];
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

  // For duck-type compatability with Tensor.  Returns a copy of the strings.
  dataSync(): string[] {
    return this.stringValues.slice();
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


  locToIndex(locs: number[]): number {
    if (this.rank === 0) {
      return 0;
    } else if (this.rank === 1) {
      return locs[0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  }

  /**
   * Sets the value in the `StringTensor` at the provided location.
   *
   * @param value The string to put into the `StringTensor`.
   * @param locs The location indices.
   */
  set(value: string, ...locs: number[]) {
    if (locs.length === 0) {
      locs = [0];
    }
    util.assert(
        locs.length === this.rank,
        `The number of provided coordinates (${locs.length}) must ` +
            `match the rank (${this.rank})`);
    const index = this.locToIndex(locs);
    this.stringValues[index] = value;
  }

  /**
   * Flatten a StringTensor to a 1D array.
   * @doc {heading: 'Tensors', subheading: 'Classes' }
   */
  flatten(): StringTensor1D {
    return this.as1D();
  }

  /**
   * Converts a size-1 `StringTensor` to a `StringScalar`.
   * @doc {heading: 'Tensors', subheading: 'Classes' }
   */
  asScalar(): StringScalar {
    util.assert(this.size === 1, 'The array must have only 1 element.');
    return this.reshape<Rank.R0>([]);
  }

  /**
   * Converts a `StringTensor` to a `StringTensor1D`.
   * @doc {heading: 'Tensors', subheading: 'Classes' }
   */
  as1D(): StringTensor1D {
    return this.reshape<Rank.R1>([this.size]);
  }

  /**
   * Converts a `StringTensor` to a `StringTensor2D`.
   *
   * @param rows Number of rows in `StringTensor2D`.
   * @param columns Number of columns in `StringTensor2D`.
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  as2D(rows: number, columns: number): StringTensor2D {
    return this.reshape<Rank.R2>([rows, columns]);
  }

  /**
   * Converts a `StringTensor` to a `StringTensor3D`.
   *
   * @param rows Number of rows in `StringTensor3D`.
   * @param columns Number of columns in `StringTensor3D`.
   * @param depth Depth of `StringTensor3D`.
   * @doc {heading: 'Tensors', subheading: 'Classes' }
   */
  as3D(rows: number, columns: number, depth: number): StringTensor3D {
    return this.reshape<Rank.R3>([rows, columns, depth]);
  }

  /**
   * Converts a `StringTensor` to a `StringTensor4D`.
   *
   * @param rows Number of rows in `StringTensor4D`.
   * @param columns Number of columns in `StringTensor4D`.
   * @param depth Depth of `StringTensor4D`.
   * @param depth2 4th dimension of `StringTensor4D`.
   * @doc {heading: 'Tensors', subheading: 'Classes' }
   */
  as4D(rows: number, columns: number, depth: number, depth2: number):
      StringTensor4D {
    return this.reshape<Rank.R4>([rows, columns, depth, depth2]);
  }

  /**
   * Reshapes the stringTensor into the provided shape.
   *
   * @param newShape An array of integers defining the output tensor shape.
   * @doc {heading: 'Tensors', subheading: 'Classes' }
   */
  reshape<R extends Rank>(newShape: ShapeMap[R]): StringTensor<R> {
    // TODO(bileschi): This implementation makes a copy of the strings, there
    // may be a way to save memory here.
    return StringTensor.make(newShape, this.stringValues);
  }

  // TODO(bileschi): Fuse with version in @tensorflow/tfjs-core slice_util.ts
  assertParamsValid(begin: number[], size: number[]): void {
    util.assert(
        this.rank === begin.length,
        `Error in slice${this.rank}D: Length of begin ${begin} must ` +
            `match the rank of the array (${this.rank}).`);
    util.assert(
        this.rank === size.length,
        `Error in slice${this.rank}D: Length of size ${size} must ` +
            `match the rank of the array (${this.rank}).`);

    for (let i = 0; i < this.rank; ++i) {
      util.assert(
          begin[i] + size[i] <= this.shape[i],
          `Error in slice${this.rank}D: begin[${i}] + size[${i}] ` +
              `(${begin[i] + size[i]}) would overflow this.shape[${i}] (${
                  this.shape[i]})`);
    }
  }

  sliceAlongOneAxis<T extends StringTensor<R>>(
      axis: number, begin: number, size: number): T {
    const _begin: number[] = [];
    const _size: number[] = [];
    if (axis < 0) {
      throw new ValueError(`Asked to slice on axis ${axis} less than 0.`);
    }
    if (this.rank <= axis) {
      throw new ValueError(`Asked to slice on axis ${axis} of a rank ${
          this.rank} stringTensor.`);
    }
    // For rank 4, axis 2, with shape S
    // _begin will be like:
    //    [0, 0, begin, 0];
    // _size will be like:
    //    [this.shape[0],
    //     this.shape[1],
    //     size,
    //     this.shape[3]];
    for (let i = 0; i < this.rank; i++) {
      if (i === axis) {
        _begin.push(begin);
        _size.push(size);
      } else {
        _begin.push(0);
        _size.push(this.shape[i]);
      }
    }
    return this.slice(_begin, _size);
  }

  // Mostly cribbed from @tensorflow/tfjs-core slice.ts
  slice<T extends StringTensor<R>>(
      begin: number|number[], size?: number|number[]): T {
    if (this.rank === 0) {
      throw new Error('Slicing stringScalar is not possible');
    }
    // The following logic allows for more ergonomic calls.
    let begin_: number[];
    if (typeof begin === 'number') {
      begin_ = [begin, ...new Array(this.rank - 1).fill(0)];
    } else if (begin.length < this.rank) {
      begin_ = begin.concat(new Array(this.rank - begin.length).fill(0));
    } else {
      begin_ = begin;
    }
    let size_: number[];
    if (size == null) {
      size_ = new Array(this.rank).fill(-1);
    } else if (typeof size === 'number') {
      size_ = [size, ...new Array(this.rank - 1).fill(-1)];
    } else if (size.length < this.rank) {
      size_ = size.concat(new Array(this.rank - size.length).fill(-1));
    } else {
      size_ = size;
    }
    size_ = size_.map((d, i) => {
      if (d >= 0) {
        return d;
      } else {
        util.assert(d === -1, 'Bad value in size');
        return this.shape[i] - begin_[i];
      }
    });
    this.assertParamsValid(begin_, size_);

    const buffer = new StringTensor<R>(size_ as ShapeMap[R]);
    for (let i = 0; i < buffer.size; ++i) {
      const loc = buffer.indexToLoc(i);
      const xLoc = loc.map((idx, j) => idx + begin_[j]);
      buffer.set(this.get(...xLoc), ...loc);
    }
    return buffer as T;
  }

  indexToLoc(index: number): number[] {
    if (this.rank === 0) {
      return [];
    } else if (this.rank === 1) {
      return [index];
    }
    const locs: number[] = new Array(this.shape.length);
    for (let i = 0; i < locs.length - 1; ++i) {
      locs[i] = Math.floor(index / this.strides[i]);
      index -= locs[i] * this.strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
  }

  // TODO(bileschi): Does dispose for string tensor need to do something more
  // than no-op?
  dispose(): void {}
}

export type StringScalar = StringTensor<Rank.R0>;
export type StringTensor1D = StringTensor<Rank.R1>;
export type StringTensor2D = StringTensor<Rank.R2>;
export type StringTensor3D = StringTensor<Rank.R3>;
export type StringTensor4D = StringTensor<Rank.R4>;
export type StringTensor5D = StringTensor<Rank.R5>;
export type StringTensor6D = StringTensor<Rank.R6>;

export class PreprocessingExports {
  /* @doc {heading: 'Tensors', subheading: 'Creation'} */
  static stringTensor1d(values: string[]): StringTensor1D {
    assertNonNull(values);
    const inferredShape = inferShape(values);
    if (inferredShape.length !== 1) {
      throw new Error('stringTensor1d() requires stringValues to be string[]');
    }
    return PreprocessingExports.stringTensor(values, inferredShape as [number]);
  }


  /* @doc {heading: 'Tensors', subheading: 'Creation'} */
  static stringTensor2d(values: string[]|string[][], shape?: [number, number]):
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
          'stringTensor2d() requires shape to be provided when `values` ' +
          'are a flat array');
    }
    shape = shape || inferredShape as [number, number];
    return PreprocessingExports.stringTensor(values, shape);
  }

  /* @doc {heading: 'Tensors', subheading: 'Creation'} */
  static stringTensor3d(values: string[]|string[][][], shape?: [
    number, number, number
  ]): StringTensor3D {
    assertNonNull(values);
    if (shape != null && shape.length !== 3) {
      throw new Error('stringTensor3d() requires shape to have three numbers');
    }
    const inferredShape = inferShape(values);
    if (inferredShape.length !== 3 && inferredShape.length !== 1) {
      throw new Error(
          'stringTensor3d() requires stringValues to be string[][][] or ' +
          'string[]');
    }
    if (inferredShape.length === 1 && shape == null) {
      throw new Error(
          'stringTensor3d() requires shape to be provided when `values` ' +
          'are a flat array');
    }
    shape = shape || inferredShape as [number, number, number];
    return PreprocessingExports.stringTensor(values, shape);
  }

  /* @doc {heading: 'Tensors', subheading: 'Creation'} */
  static stringTensor4d(values: string[]|string[][][][], shape?: [
    number, number, number, number
  ]): StringTensor4D {
    assertNonNull(values);
    if (shape != null && shape.length !== 4) {
      throw new Error('stringTensor4d() requires shape to have four numbers');
    }
    const inferredShape = inferShape(values);
    if (inferredShape.length !== 4 && inferredShape.length !== 1) {
      throw new Error(
          'stringTensor4d() requires stringValues to be string[][][][] or ' +
          'string[]');
    }
    if (inferredShape.length === 1 && shape == null) {
      throw new Error(
          'stringTensor4d() requires shape to be provided when `values` ' +
          'are a flat array');
    }
    shape = shape || inferredShape as [number, number, number, number];
    return PreprocessingExports.stringTensor(values, shape);
  }

  /* @doc {heading: 'Tensors', subheading: 'Creation'} */
  static stringTensor5d(values: string[]|string[][][][][], shape?: [
    number, number, number, number, number
  ]): StringTensor5D {
    assertNonNull(values);
    if (shape != null && shape.length !== 5) {
      throw new Error('stringTenso5d() requires shape to have five numbers');
    }
    const inferredShape = inferShape(values);
    if (inferredShape.length !== 5 && inferredShape.length !== 1) {
      throw new Error(
          'stringTensor5d() requires stringValues to be string[][][][][] or ' +
          'string[]');
    }
    if (inferredShape.length === 1 && shape == null) {
      throw new Error(
          'stringTensor5d() requires shape to be provided when `values` ' +
          'are a flat array');
    }
    shape = shape || inferredShape as [number, number, number, number, number];
    return PreprocessingExports.stringTensor(values, shape);
  }

  /* @doc {heading: 'Tensors', subheading: 'Creation'} */
  static stringTensor6d(values: string[]|string[][][][][][], shape?: [
    number, number, number, number, number, number
  ]): StringTensor6D {
    assertNonNull(values);
    if (shape != null && shape.length !== 6) {
      throw new Error('stringTensor6d() requires shape to have six numbers');
    }
    const inferredShape = inferShape(values);
    if (inferredShape.length !== 6 && inferredShape.length !== 1) {
      throw new Error(
          'stringTensor6d() requires stringValues to be string[][][][][][] ' +
          'or string[]');
    }
    if (inferredShape.length === 1 && shape == null) {
      throw new Error(
          'stringTensor6d() requires shape to be provided when `values` ' +
          'are a flat array');
    }
    shape = shape ||
        inferredShape as [number, number, number, number, number, number];
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
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
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
