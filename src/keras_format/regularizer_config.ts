/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

export interface L1L2PrimitiveArgs {
  /** L1 regularization rate. Defaults to 0.01. */
  l1?: number;
  /** L2 regularization rate. Defaults to 0.01. */
  l2?: number;
}

export type L1L2Config = L1L2PrimitiveArgs;

export interface L1PrimitiveArgs {
  /** L1 regularization rate. Defaults to 0.01. */
  l1: number;
}

export type L1Config = L1PrimitiveArgs;

export interface L2PrimitiveArgs {
  /** L2 regularization rate. Defaults to 0.01. */
  l2: number;
}

export type L2Config = L2PrimitiveArgs;

export type RegularizerConfig = L1L2Config|L1Config|L2Config;
