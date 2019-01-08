/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

export interface MaxNormPrimitiveArgs {
  /**
   * Maximum norm for incoming weights
   */
  maxValue?: number;
  /**
   * Axis along which to calculate norms.
   *
   *  For instance, in a `Dense` layer the weight matrix
   *  has shape `[inputDim, outputDim]`,
   *  set `axis` to `0` to constrain each weight vector
   *  of length `[inputDim,]`.
   *  In a `Conv2D` layer with `dataFormat="channels_last"`,
   *  the weight tensor has shape
   *  `[rows, cols, inputDepth, outputDepth]`,
   *  set `axis` to `[0, 1, 2]`
   *  to constrain the weights of each filter tensor of size
   *  `[rows, cols, inputDepth]`.
   */
  axis?: number;
}

export type MaxNormConfig = MaxNormPrimitiveArgs;

export interface UnitNormPrimitiveArgs {
  /**
   * Axis along which to calculate norms.
   *
   * For instance, in a `Dense` layer the weight matrix
   * has shape `[inputDim, outputDim]`,
   * set `axis` to `0` to constrain each weight vector
   * of length `[inputDim,]`.
   * In a `Conv2D` layer with `dataFormat="channels_last"`,
   * the weight tensor has shape
   * [rows, cols, inputDepth, outputDepth]`,
   * set `axis` to `[0, 1, 2]`
   * to constrain the weights of each filter tensor of size
   * `[rows, cols, inputDepth]`.
   */
  axis?: number;
}

export type UnitNormConfig = UnitNormPrimitiveArgs;

export interface MinMaxNormPrimitiveArgs {
  /**
   * Minimum norm for incoming weights
   */
  minValue?: number;
  /**
   * Maximum norm for incoming weights
   */
  maxValue?: number;
  /**
   * Axis along which to calculate norms.
   * For instance, in a `Dense` layer the weight matrix
   * has shape `[inputDim, outputDim]`,
   * set `axis` to `0` to constrain each weight vector
   * of length `[inputDim,]`.
   * In a `Conv2D` layer with `dataFormat="channels_last"`,
   * the weight tensor has shape
   * `[rows, cols, inputDepth, outputDepth]`,
   * set `axis` to `[0, 1, 2]`
   * to constrain the weights of each filter tensor of size
   * `[rows, cols, inputDepth]`.
   */
  axis?: number;
  /**
   * Rate for enforcing the constraint: weights will be rescaled to yield:
   * `(1 - rate) * norm + rate * norm.clip(minValue, maxValue)`.
   * Effectively, this means that rate=1.0 stands for strict
   * enforcement of the constraint, while rate<1.0 means that
   * weights will be rescaled at each step to slowly move
   * towards a value inside the desired interval.
   */
  rate?: number;
}

export type MinMaxNormConfig = MinMaxNormPrimitiveArgs;

export type ConstraintConfig = MaxNormConfig|UnitNormConfig|MinMaxNormConfig;
