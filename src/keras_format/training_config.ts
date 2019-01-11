/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BaseSerialization, PyJsonDict} from './types';

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
 * See internal issue: b/121033602
 */
export type OptimizerConfig = PyJsonDict;

/**
 * List of all known loss names, along with a string description.
 */
export type LossOptions = {
  mean_squared_error: 'Mean Squared Error',
  mean_absolute_error: 'Mean Absolute Error',
  mean_absolute_percentage_error: 'Mean Absolute Percentage Error',
  mean_squared_logarithmic_error: 'Mean Squared Logarithmic Error',
  squared_hinge: 'Squared Hinge',
  hinge: 'Hinge',
  categorical_hinge: 'Categorical Hinge',
  logcosh: 'Logcosh',
  categorical_crossentropy: 'Categorical Cross Entropy',
  sparse_categorical_crossentropy: 'Sparse Categorical Cross Entropy',
  kullback_leibler_divergence: 'Kullback-Liebler Divergence',
  poisson: 'Poisson',
  cosine_proximity: 'Cosine Proximity',
};

export type LossKey = keyof LossOptions;

// TODO(soergel): flesh out known metrics options
export type MetricsKey = string;

/**
 * Configuration of that which is necessary to train a given model and
 * dataset. This includes the configuration to the optimizer, the loss, any
 * metrics to be calculated, etc.
 */
export interface TrainingConfig {
  optimizerConfig: OptimizerSerialization<string, PyJsonDict>;
  loss: LossKey;
  metrics: MetricsKey[];
  sampleWeightMode: string;
  lossWeights: null;  // TQDQ(soergel): wat
}
