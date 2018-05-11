/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';

import {getScopedTensorName, getUniqueTensorName} from './common';
import {Constraint} from './constraints';
import {DType, getNextUniqueTensorId, Shape, SymbolicTensor} from './types';

const DEFAULT_VARIABLE_NAME_PREFIX = 'Variable';

/**
 * A `LayerVariable` is similar to a `Tensor` in that it has a dtype and shape,
 * but its value is mutable.  The value is itself represented as a `Tensor`, and
 * can be read with the `read()` method and updated with the `write()` method.
 */
export class LayerVariable {
  readonly dtype: DType;
  readonly shape: Shape;

  readonly id: number;
  // The fully scoped name of this Variable, including a unique suffix if needed
  readonly name: string;
  // The originally requested fully scoped name of this Variable, not including
  // any unique suffix.  This may be needed when restoring weights because this
  // original name is used as a key.
  readonly originalName: string;
  readonly trainable: boolean;

  protected readonly val: tfc.Variable;
  readonly constraint: Constraint;
  /**
   * Construct Variable from a Tensor.
   *
   * If not explicitly named, the Variable will be given a name with the
   * prefix 'Variable'. Variable names are unique. In the case of name
   * collision, suffixies '_<num>' will be added to the name.
   *
   * @param val Initial value of the Variable.
   * @param name Name of the variable. If `null` or `undefined` is provided, it
   *   will default a name with the prefix 'Variable'.
   * @param constraint Optional, projection function to be applied to the
   * variable after optimize updates
   * @throws ValueError if `name` is `null` or `undefined`.
   */
  constructor(
      val: tfc.Tensor, dtype: DType = DType.float32,
      name = DEFAULT_VARIABLE_NAME_PREFIX, trainable = true,
      constraint: Constraint = null) {
    this.dtype = dtype == null ? DType.float32 : dtype;
    this.shape = val.shape;
    this.id = getNextUniqueTensorId();

    name = name == null ? DEFAULT_VARIABLE_NAME_PREFIX : name;
    this.originalName = getScopedTensorName(name);
    this.name = getUniqueTensorName(this.originalName);

    this.trainable = trainable;
    this.constraint = constraint;

    this.val = tfc.variable(val, this.trainable, this.name, this.dtype);
  }

  /**
   * Get a snapshot of the Variable's value.
   *
   * The returned value is a snapshot of the Variable's value at the time of
   * the invocation. Future mutations in the value of the tensor will only
   * be reflected by future calls to this method.
   */
  read(): tfc.Tensor {
    return this.val;
  }

  /**
   * Update the value of the Variable.
   *
   * @param newVal: The new value to update to. Must be consistent with the
   *   dtype and shape of the Variable.
   * @return This Variable.
   */
  write(newVal: tfc.Tensor) {
    // TODO(cais): Once  TF.js Core supports Tensor.dtype, check dtype match.
    checkShapesMatch(this.val, newVal);
    this.val.assign(newVal);
    if (this.constraint != null) {
      this.val.assign(this.constraint.apply(this.val));
    }
    return this;
  }
}

function checkShapesMatch(
    x: tfc.Tensor|SymbolicTensor, y: tfc.Tensor|SymbolicTensor): void {
  if (x.shape.toString() !== y.shape.toString()) {
    throw new Error(
        'Shape mismatch: ' + JSON.stringify(x.shape) + ' vs. ' +
        JSON.stringify(y.shape));
  }
}
