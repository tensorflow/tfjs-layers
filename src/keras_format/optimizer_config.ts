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
import {stringDictToArray} from './utils';

// Because of the limitations in the current Keras spec, there is no clear
// definition of what may or may not be the configuration of an optimizer.
//
// For now we'll represent the ones available in TF.js--but it will take more
// thought to get this right in a cross-platform way.
//
// See internal issue: b/121033602

// TODO(soergel): This is a stopgap that needs further thought.
// Does it belong here?
// Does it belong in tfjs-core?
// See also the dormant https://github.com/tensorflow/tfjs-core/pull/1404

export type AdadeltaOptimizerConfig = {
  learning_rate: number; rho: number; epsilon: number;
};

export type AdadeltaSerialization =
    BaseSerialization<'AdadeltaOptimizer', AdadeltaOptimizerConfig>;

export type AdagradOptimizerConfig = {
  learning_rate: number; initial_accumulator_value: number;
};

export type AdagradSerialization =
    BaseSerialization<'AdagradOptimizer', AdagradOptimizerConfig>;

export type AdamOptimizerConfig = {
  learning_rate: number; beta1: number; beta2: number; epsilon: number;
};

export type AdamSerialization =
    BaseSerialization<'AdamOptimizer', AdamOptimizerConfig>;

export type AdamaxOptimizerConfig = {
  learning_rate: number; beta1: number; beta2: number; epsilon: number;
  decay?: number;
};

export type AdamaxSerialization =
    BaseSerialization<'AdamaxOptimizer', AdamaxOptimizerConfig>;

export type MomentumOptimizerConfig = {
  // extends SGDOptimizerConfig {
  learning_rate: number; momentum: number; use_nesterov: boolean;
};

export type MomentumSerialization =
    BaseSerialization<'MomentumOptimizer', MomentumOptimizerConfig>;

export type RMSPropOptimizerConfig = {
  learning_rate: number;
  decay?: number;
  momentum?: number;
  epsilon?: number;
  centered?: boolean;
};

export type RMSPropSerialization =
    BaseSerialization<'RMSPropOptimizer', RMSPropOptimizerConfig>;

export type SGDOptimizerConfig = {
  learning_rate: number;
};

export type SGDSerialization =
    BaseSerialization<'SGDOptimizer', SGDOptimizerConfig>;

// Update NUM_OPTIMIZER_OPTIONS in concert (for testing)
export type OptimizerSerialization = AdadeltaSerialization|AdagradSerialization|
    AdamSerialization|AdamaxSerialization|MomentumSerialization|
    RMSPropSerialization|SGDSerialization;

export const NUM_OPTIMIZER_OPTIONS = 7;

export type OptimizerClassName = OptimizerSerialization['class_name'];

// This helps guarantee that the Options class below is complete.
export type OptimizerOptionMap = {
  [key in OptimizerClassName]: string
};

/**
 * List of all known optimizer names, along with a string description.
 *
 * Representing this as a class allows both type-checking using the keys and
 * automatically translating to human readable display names where needed.
 */
class OptimizerOptions implements OptimizerOptionMap {
  [key: string]: string;
  // tslint:disable:variable-name
  public readonly AdadeltaOptimizer = 'Adadelta';
  public readonly AdagradOptimizer = 'Adagrad';
  public readonly AdamOptimizer = 'Adam';
  public readonly AdamaxOptimizer = 'Adamax';
  public readonly MomentumOptimizer = 'Momentum';
  public readonly RMSPropOptimizer = 'RMSProp';
  public readonly SGDOptimizer = 'SGD';
  // tslint:enable:variable-name
}

/**
 * An array of `{value, label}` pairs describing the valid optimizers.
 *
 * The `value` is the serializable string constant, and the `label` is a more
 * user-friendly description (e.g. for use in UIs).
 */
export const optimizerOptions = stringDictToArray(new OptimizerOptions());

/**
 * A string array of valid Optimizer class names.
 *
 * This is guaranteed to match the `OptimizerClassName` union type.
 */
export const optimizerClassNames = optimizerOptions.map((x) => x.value);
