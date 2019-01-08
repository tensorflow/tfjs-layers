import {DataType} from '@tensorflow/tfjs-core';

import {Shape} from '../types';

/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/** Constructor arguments for Layer. */
export interface LayerPrimitiveArgs {
  /**
   * If defined, will be used to create an input layer to insert before this
   * layer. If both `inputShape` and `batchInputShape` are defined,
   * `batchInputShape` will be used. This argument is only applicable to input
   * layers (the first layer of a model).
   */
  inputShape?: Shape;
  /**
   * If defined, will be used to create an input layer to insert before this
   * layer. If both `inputShape` and `batchInputShape` are defined,
   * `batchInputShape` will be used. This argument is only applicable to input
   * layers (the first layer of a model).
   */
  batchInputShape?: Shape;
  /**
   * If `inputShape` is specified and `batchInputShape` is *not* specifiedd,
   * `batchSize` is used to construct the `batchInputShape`: `[batchSize,
   * ...inputShape]`
   */
  batchSize?: number;
  /**
   * The data-type for this layer. Defaults to 'float32'.
   * This argument is only applicable to input layers (the first layer of a
   * model).
   */
  dtype?: DataType;
  /** Name for this layer. */
  name?: string;
  /** Whether this layer is trainable. Defaults to true. */
  trainable?: boolean;
  /** Whether the weights of this layer are updatable by `fit`. */
  updatable?: boolean;
  /** Legacy support. Do not use for new code. */
  inputDType?: DataType;
}

export type LayerConfig = LayerPrimitiveArgs;
