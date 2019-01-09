/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

export interface L1L2BaseConfig {
  /** L1 regularization rate. Defaults to 0.01. */
  l1?: number;
  /** L2 regularization rate. Defaults to 0.01. */
  l2?: number;
}

export type L1L2Config = L1L2BaseConfig;

export interface L1L2Serialization {
  class_name: 'l1_l2';
  config: L1L2Config;
}

export interface L1BaseConfig {
  /** L1 regularization rate. Defaults to 0.01. */
  l1: number;
}

export type L1Config = L1BaseConfig;

export interface L1Serialization {
  class_name: 'l1';
  config: L1Config;
}

export interface L2BaseConfig {
  /** L2 regularization rate. Defaults to 0.01. */
  l2: number;
}

export type L2Config = L2BaseConfig;

export interface L2Serialization {
  class_name: 'l2';
  config: L2Config;
}

export type RegularizerSerialization =
    L1L2Serialization|L1Serialization|L2Serialization;
