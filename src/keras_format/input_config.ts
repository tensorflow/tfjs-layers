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

import {Shape} from './common';
import {BaseSerialization} from './types';

export type InputLayerConfig = {
  inputShape?: Shape;
  batchSize?: number;
  batchInputShape?: Shape;
  dtype?: DataType;
  sparse?: boolean;
  name?: string;
};

export type InputLayerSerialization =
    BaseSerialization<'Input', InputLayerConfig>;
