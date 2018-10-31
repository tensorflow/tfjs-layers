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
 * Executor: Evaluates SymbolicTensor based on feeds.
 */

import {cast, dispose, Tensor} from '@tensorflow/tfjs-core';

import {ValueError} from '../errors';
import {Kwargs} from '../types';
import {toList} from '../utils/generic_utils';

import {InputLayer} from './input_layer';
import {SymbolicTensor} from './topology';

/**
 * Helper function to check the dtype and shape compatibility of a feed value.
 */
function assertFeedCompatibility(key: SymbolicTensor, val: Tensor): Tensor {
  // 1. Check shape compatibility.  If shapes are not compatible, error.
  if (key.shape != null) {
    if (key.shape.length !== val.shape.length) {
      throw new ValueError(
          `The rank of feed (${val.shape.length}) does not match the rank of ` +
          `the key (${key.shape.length}).`);
    }

    for (let i = 0; i < key.shape.length; ++i) {
      if (key.shape[i] != null && key.shape[i] !== val.shape[i]) {
        throw new ValueError(
            `The ${i}-th dimension of the feed (${val.shape[i]}) is ` +
            `incompatible with that of the key (${key.shape[i]}).`);
      }
    }
  }
  // 2. Check dtype compatibility.
  if (key.dtype == null || key.dtype === val.dtype) {
    //  2a.  If types match, return val tensor as is.
    return val;
  }
  try {
    //  2b. Attempt to convert to expected type.
    return cast(val, key.dtype);
  } catch (err) {
    //  2c. If conversion fails, return helpful error.
    throw new ValueError(
        `The dtype of the feed (${val.dtype}) can not be cast to the dtype ` +
        `of the key '${key.name}' (${key.dtype}).`);
  }
}

/**
 * A concrete Tensor value for a symbolic tensor as the key.
 */
export interface Feed {
  key: SymbolicTensor;
  value: Tensor;
}

/**
 * FeedDict: A mapping from unique SymbolicTensors to feed values for them.
 * A feed value is a concrete value represented as an `Tensor`.
 */
export class FeedDict {
  private id2Value: {[id: number]: Tensor} = {};
  private id2Name: {[id: number]: string} = {};

  /**
   * Constructor, optionally does copy-construction.
   * @param feeds An Array of `Feed`s, or another `FeedDict`, in which case
   *   copy-construction will be performed.
   */
  constructor(feeds?: Feed[]|FeedDict) {
    if (feeds instanceof FeedDict) {
      for (const id in feeds.id2Value) {
        this.id2Value[id] = feeds.id2Value[id];
      }
    } else {
      if (feeds == null) {
        return;
      }
      for (const feed of feeds) {
        this.add(feed.key, feed.value);
      }
    }
  }

  /**
   * Add a key-value pair to the FeedDict.
   * @param key The key of the feed.
   * @param value The value of the feed.
   * @returns This `FeedDict`.
   * @throws ValueError: If the key `SymbolicTensor` already exists in the
   *   `FeedDict`.
   */
  add(key: SymbolicTensor, value: Tensor): FeedDict {
    if (this.id2Value[key.id] == null) {
      this.id2Value[key.id] = assertFeedCompatibility(key, value);
      this.id2Name[key.id] = key.name;
    } else {
      throw new ValueError(`Duplicate key: name=${key.name}, id=${key.id}`);
    }
    return this;
  }

  /**
   * Add a Feed to the FeedDict.
   * @param feed The new `Feed` to add.
   * @returns This `FeedDict`.
   */
  addFeed(feed: Feed) {
    this.add(feed.key, feed.value);
  }

  /**
   * Probe whether a key already exists in the FeedDict.
   * @param key
   */
  hasKey(key: SymbolicTensor): boolean {
    return this.id2Value[key.id] != null;
  }

  hasName(name: string): boolean {
    for (const id of Object.keys(this.id2Value)) {
      if (this.id2Name[+id] === name) {
        return true;
      }
    }
    return false;
  }

  getValueByName(name: string): Tensor {
    for (const id of Object.keys(this.id2Value)) {
      if (this.id2Name[+id] === name) {
        return this.id2Value[+id];
      }
    }
    throw new ValueError(`Nonexistent name: ${name}`);
  }

  /**
   * Get all the SymbolicTensor available in this FeedDict.
   */
  names() {
    return Object.keys(this.id2Value).map(id => this.id2Name[+id]);
  }

  /**
   * Get the feed value for given key.
   * @param key
   * @returns If `key` exists, the corresponding feed value.
   * @throws ValueError: If `key` does not exist in this `FeedDict`.
   */
  getValue(key: SymbolicTensor): Tensor {
    if (this.id2Value[key.id] == null) {
      throw new ValueError(`Nonexistent key: ${key.name}`);
    } else {
      return this.id2Value[key.id];
    }
  }
}

/**
 * Execute a SymbolicTensor by using concrete feed values.
 *
 * A `SymbolicTensor` object is a node in a computation graph of TF.js
 * Layers. The object is backed by a source layer and input
 * `SymbolicTensor`s to the source layer. This method evaluates
 * the `call()` method of the source layer, using concrete values of the inputs
 * obtained from either
 * * `feedDict`, if the input key exists in `feedDict`, or else,
 * * a recursive call to `execute()` itself.
 *
 * @param x: The `SymbolicTensor` to execute.
 * @param feedDict: The feed values, as base condition of the recursion.
 *   execution.
 * @param kwargs: Optional keyword arguments.
 * @returns Result of the execution.
 * @throws ValueError: If any `SymbolicTensor`s from `InputLayer`s
 *   encountered during the execution lacks a feed value in `feedDict`.
 */
export function execute(
    fetches: SymbolicTensor|SymbolicTensor[], feedDict: FeedDict,
    kwargs?: Kwargs): Tensor|Tensor[]|[Tensor | Tensor[]] {
  const training: boolean = kwargs == null ? false : kwargs['training'];

  const arrayFetches = Array.isArray(fetches);
  const fetchArray: SymbolicTensor[] =
      arrayFetches ? fetches as SymbolicTensor[] : [fetches as SymbolicTensor];

  const outputNames = fetchArray.map(t => t.name);
  const finalOutputs: Tensor[] = [];
  for (const outputName of outputNames) {
    if (feedDict.hasName(outputName)) {
      finalOutputs.push(feedDict.getValueByName(outputName));
    } else {
      finalOutputs.push(null);
    }
  }

  const visited = new Set<string>();
  // Put keys of the feedDict into visited first, so they don't have to be
  // walked. This is useful in case where there are feeds for intermediate
  // SymbolicTensors of the graph.
  for (const key of feedDict.names()) {
    visited.add(key);
  }
  const sorted: SymbolicTensor[] = [];
  const recipientCounts: {[fetchName: string]: number} = {};
  getTpologicalSortAndRecipientMap(
      fetchArray, sorted, recipientCounts, visited);
  visited.clear();  // For memory savings.

  const internalFeedDict = new FeedDict(feedDict);

  for (let i = 0; i < sorted.length; ++i) {
    const symbolic = sorted[i];
    if (symbolic.sourceLayer instanceof InputLayer) {
      continue;
    }
    const inputValues: Tensor[] = [];
    const tensorsToDispose: Tensor[] = [];
    for (const input of symbolic.inputs) {
      const value = internalFeedDict.getValue(input);
      inputValues.push(value);
      recipientCounts[input.name]--;
      if (recipientCounts[input.name] === 0 && !feedDict.hasKey(input) &&
          outputNames.indexOf(input.name) === -1 && !value.isDisposed) {
        tensorsToDispose.push(value);
      }
    }
    const output =
        toList(symbolic.sourceLayer.apply(inputValues, kwargs)) as Tensor[];
    const layerOutputs = getNodeOutputs(symbolic);
    const outputSymbolicTensors =
        Array.isArray(layerOutputs) ? layerOutputs : [layerOutputs];
    for (let i = 0; i < outputSymbolicTensors.length; ++i) {
      if (!internalFeedDict.hasKey(outputSymbolicTensors[i])) {
        internalFeedDict.add(outputSymbolicTensors[i], output[i]);
      }
      const index = outputNames.indexOf(outputSymbolicTensors[i].name);
      if (index !== -1) {
        finalOutputs[index] = output[i];
      }
    }

    if (!training) {
      // console.log(`# Disposing ${tensorsToDispose.length}`);  // DEBUG
      dispose(tensorsToDispose);
    }
  }

  return arrayFetches ? finalOutputs : finalOutputs[0];
}

/**
 * Use depth-first search (DFS) to sort the `SymbolicTensor`s topologically.
 *
 * @param fetch
 * @param sorted
 * @param visited
 */
function getTpologicalSortAndRecipientMap(
    fetches: SymbolicTensor[], sorted: SymbolicTensor[],
    recipientCounts: {[fetchName: string]: number}, visited: Set<string>) {
  const fetchSortedArrays: SymbolicTensor[][] = [];

  for (const fetch of fetches) {
    const fetchSorted: SymbolicTensor[] = [];
    if (visited.has(fetch.name)) {
      continue;
    }
    visited.add(fetch.name);
    fetchSorted.push(fetch);
    if (fetch.inputs.length > 0) {
      for (const input of fetch.inputs) {
        if (recipientCounts[input.name] == null) {
          recipientCounts[input.name] = 1;
        } else {
          recipientCounts[input.name]++;
        }
      }
      // Recursive call.
      getTpologicalSortAndRecipientMap(
          fetch.inputs, sorted, recipientCounts, visited);
    }
    fetchSortedArrays.push(fetchSorted);
  }
  for (const fetchSorted of fetchSortedArrays) {
    while (fetchSorted.length > 0) {
      sorted.push(fetchSorted.splice(0, 1)[0]);
    }
  }
}

/**
 * Get the symbolic output tensors of the node to which a given fetch belongs.
 * @param fetch The fetched symbolic tensor.
 * @returns The Array of symbolic tensors output by the node to which `fetch`
 *   belongs.
 */
function getNodeOutputs(fetch: SymbolicTensor): SymbolicTensor|
    SymbolicTensor[] {
  let layerOutputs: SymbolicTensor|SymbolicTensor[];
  if (fetch.sourceLayer.inboundNodes.length === 1) {
    layerOutputs = fetch.sourceLayer.output;
  } else {
    let nodeIndex: number = null;
    for (let i = 0; i < fetch.sourceLayer.inboundNodes.length; ++i) {
      for (const outputTensor of fetch.sourceLayer.inboundNodes[i]
               .outputTensors) {
        if (outputTensor.id === fetch.id) {
          nodeIndex = i;
          break;
        }
      }
    }
    layerOutputs = fetch.sourceLayer.getOutputAt(nodeIndex);
  }
  return layerOutputs;
}
