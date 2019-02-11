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
 * Stub interfaces and classes for testing tf.Model.fitDataset().
 *
 * TODO(cais, soergel): Remove this in favor of actual interfaces and classes
 *   when ready.
 */

export abstract class LazyIterator<T> {
  abstract async next(): Promise<IteratorResult<T>>;
}

export abstract class Dataset<T> {
  abstract async iterator(): Promise<LazyIterator<T>>;
  size: number;
}
