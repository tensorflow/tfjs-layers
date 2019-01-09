/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataType} from '@tensorflow/tfjs-core';

import {Shape} from '../types';

/**
 * Constructor arguments for InputLayer.
 *
 * Note: You should provide only inputShape or batchInputShape (not both).
 * If only inputShape is provided, then the batchInputShape is determined by
 * the batchSize argument and the inputShape: [batchSize].concat(inputShape).
 */
export interface InputLayerBaseConfig {
  /** Input shape, not including the batch axis. */
  inputShape?: Shape;
  /** Optional input batch size (integer or null). */
  batchSize?: number;
  /** Batch input shape, including the batch axis. */
  batchInputShape?: Shape;
  /** Datatype of the input.  */
  dtype?: DataType;
  /**
   * Whether the placeholder created is meant to be sparse.
   */
  sparse?: boolean;  // TODO(michaelterry): Not clear whether we'll need this.

  /** Name of the layer. */
  name?: string;
}

export type InputLayerConfig = InputLayerBaseConfig;

export interface InputLayerSerialization {
  class_name: 'Input';
  config: InputLayerConfig;
}
