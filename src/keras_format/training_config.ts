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
 * Configuration of that which is necessary to train a given model and dataset.
 * This includes the configuration to the optimizer, the loss, any metrics
 * to be calculated, etc.
 */

import {BaseSerialization, PyJsonDict} from './types';

export interface TrainingConfig {
  optimizerConfig: OptimizerSerialization<string, PyJsonDict>;
  loss: string;
  metrics: string[];
  sampleWeightMode: string;
  lossWeights: null;
}

/**
 * Configuration of a Keras optimizer, containing both the type of the optimizer
 * and the configuration for the optimizer of that type.
 */
export type OptimizerSerialization<N extends string,
                                             C extends OptimizerConfig> =
    BaseSerialization<N, C>;

/**
 * Because of the limitations in the current Keras spec, there is no clear
 * definition of what may or may not be the configuration of an optimizer.  Thus
 * this empty interface represents a stopgap in the Keras spec.
 *
 * See internal issue : b/121033602
 */
export type OptimizerConfig = PyJsonDict;

/**
 * List of all known loss names, along with a string description.
 */
export const tfLossOptions = [
  {label: 'Mean Squared Error', value: 'mean_squared_error'},
  {label: 'Mean Absolute Error', value: 'mean_absolute_error'}, {
    label: 'Mean Absolute Percentage Error',
    value: 'mean_absolute_percentage_error',
  },
  {
    label: 'Mean Squared Logarithmic Error',
    value: 'mean_squared_logarithmic_error',
  },
  {label: 'Squared Hinge', value: 'squared_hinge'},
  {label: 'Hinge', value: 'hinge'},
  {label: 'Categorical Hinge', value: 'categorical_hinge'},
  {label: 'Logcosh', value: 'logcosh'},
  {label: 'Categorical Cross Entropy', value: 'categorical_crossentropy'}, {
    label: 'Sparse Categorical Cross Entropy',
    value: 'sparse_categorical_crossentropy',
  },
  {label: 'Binary Cross Entropy', value: 'binary_crossentropy'}, {
    label: 'Kullback-Liebler Divergence',
    value: 'kullback_leibler_divergence',
  },
  {label: 'Poisson', value: 'poisson'},
  {label: 'Cosine Proximity', value: 'cosine_proximity'}
];
