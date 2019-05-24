/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import {Tensor} from '@tensorflow/tfjs-core';

import * as losses from './losses';
import * as metrics from './metrics';

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'binaryAccuracy'
 * }
 */
export function binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.binaryAccuracy(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'binaryCrossentropy'
 * }
 */
export function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.binaryCrossentropy(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'sparseCategoricalAccuracy'
 * }
 */
export function sparseCategoricalAccuracy(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.sparseCategoricalAccuracy(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'categoricalAccuracy'
 * }
 */
export function categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.categoricalAccuracy(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'categoricalCrossentropy'
 * }
 */
export function categoricalCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.categoricalCrossentropy(yTrue, yPred);
}

/**
 * Top K categorical accuracy metric function.
 *
 * Example:
 * ```js
 * const yTrue = tf.tensor2d([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]);
 * const yPred = tf.tensor2d([[0, 1, 0], [1, 0, 0]]);
 * const k = 2;
 * const accuracy = tf.metrics.topKCategoricalAccuracy(yTrue, yPred, k);
 * accuracy.print();
 * ```
 *
 * @param yTrue True values.
 * @param yPred Predicted values.
 * @param k Optional Number of top elements to look at for computing metrics,
 *    default to 5.
 * @returns Accuracy tensor.
 */
/** @doc {heading: 'Metrics', namespace: 'metrics'} */
export function topKCategoricalAccuracy(
  yTrue: Tensor, yPred: Tensor, k?: number): Tensor {
  return metrics.topKCategoricalAccuracy(yTrue, yPred, k);
}

/**
 * Top K sparse categorical accuracy metric function.
 *
 * Example:
 * ```js
 * const yTrue = tf.tensor1d([1, 0]);
 * const yPred = tf.tensor2d([[0, 1, 0], [1, 0, 0]]);
 * const k = 2;
 * const accuracy = tf.metrics.sparseTopKCategoricalAccuracy(yTrue, yPred, k);
 * accuracy.print();
 * ```
 *
 * @param yTrue True labels: indices.
 * @param yPred Predicted values.
 * @param k Optional Number of top elements to look at for computing metrics,
 *    default to 5.
 * @returns Accuracy tensor.
 */
/** @doc {heading: 'Metrics', namespace: 'metrics'} */
export function sparseTopKCategoricalAccuracy(
  yTrue: Tensor, yPred: Tensor, k?: number): Tensor {
  return metrics.sparseTopKCategoricalAccuracy(yTrue, yPred, k);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'precision'
 * }
 */
export function precision(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.precision(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'recall'
 * }
 */
export function recall(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.recall(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'cosineProximity'
 * }
 */
export function cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.cosineProximity(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'meanAbsoluteError'
 * }
 */
export function meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanAbsoluteError(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'meanAbsolutePercentageError'
 * }
 */
export function meanAbsolutePercentageError(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanAbsolutePercentageError(yTrue, yPred);
}

export function MAPE(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanAbsolutePercentageError(yTrue, yPred);
}

export function mape(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanAbsolutePercentageError(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'meanSquaredError'
 * }
 */
export function meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanSquaredError(yTrue, yPred);
}

export function MSE(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanSquaredError(yTrue, yPred);
}

export function mse(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanSquaredError(yTrue, yPred);
}
