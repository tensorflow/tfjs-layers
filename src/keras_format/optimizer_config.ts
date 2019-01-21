/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BaseSerialization} from './types';

// TODO(soergel): This is a stopgap that needs further thought.
// Does it belong here?
// Does it belong in tfjs-core?
// See https://github.com/tensorflow/tfjs-core/pull/1404

export type AdadeltaOptimizerConfig = {
  learningRate: number; rho: number; epsilon: number;
};

export type AdadeltaSerialization =
    BaseSerialization<'Adadelta', AdadeltaOptimizerConfig>;

export type AdagradOptimizerConfig = {
  learningRate: number; initialAccumulatorValue: number;
};

export type AdagradSerialization =
    BaseSerialization<'Adadelta', AdagradOptimizerConfig>;

export type AdamOptimizerConfig = {
  learningRate: number; beta1: number; beta2: number; epsilon: number;
};

export type AdamSerialization =
    BaseSerialization<'Adadelta', AdamOptimizerConfig>;

export type AdamaxOptimizerConfig = {
  learningRate: number; beta1: number; beta2: number; epsilon: number;
  decay?: number;
};

export type AdamaxSerialization =
    BaseSerialization<'Adadelta', AdamaxOptimizerConfig>;

export type MomentumOptimizerConfig = {
  // extends SGDOptimizerConfig {
  learningRate: number; momentum: number; useNesterov: boolean;
};

export type MomentumSerialization =
    BaseSerialization<'Momentum', MomentumOptimizerConfig>;

export type RMSPropOptimizerConfig = {
  learningRate: number;
  decay?: number;
  momentum?: number;
  epsilon?: number;
  centered?: boolean;
};

export type RMSPropSerialization =
    BaseSerialization<'RMSProp', RMSPropOptimizerConfig>;

export type SGDOptimizerConfig = {
  learningRate: number;
};

export type SGDSerialization = BaseSerialization<'SGD', SGDOptimizerConfig>;

export type OptimizerSerialization = AdadeltaSerialization|AdagradSerialization|
    AdamSerialization|AdamaxSerialization|MomentumSerialization|
    RMSPropSerialization|SGDSerialization;

export type OptimizerClassName = OptimizerSerialization['class_name'];
