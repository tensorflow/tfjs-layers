import {LayerConfig} from './engine/topology';
import {RegularizerConfig} from './regularizers';

/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * The type of a layer represented as a string, (known as class_name in Keras).
 */
export type LayerName = string;

/**
 * The index of a Node, identifying a specific invocation of a given Layer.
 */
export type NodeIndex = number;

/**
 * The index of a Tensor output by a given Node of a given Layer.
 */
export type TensorIndex = number;

/**
 * Arguments to the apply(...) method that produced a specific Node.
 */
// tslint:disable-next-line:no-empty-interface
export interface NodeArgs {}

/**
 * A reference to a specific Tensor, given by its Layer name, Node index, and
 * output index, including the apply() arguments associated with the Node.
 *
 * This is used in `NodeConfig` to specify the inputs to each Node.
 */
export type TensorKeyWithArgsArray =
    [LayerName, NodeIndex, TensorIndex, NodeArgs];

// TODO(soergel): verify behavior of Python Keras; maybe PR to standardize it.
/**
 * A reference to a specific Tensor, given by its Layer name, Node index, and
 * output index.
 *
 * This does not include the apply() arguments associated with the Node.  It is
 * used in the Model config to specify the inputLayers and outputLayers.  It
 * seems to be an idiosyncrasy of Python Keras that the node arguments are not
 * included here.
 */
export type TensorKeyArray = [LayerName, NodeIndex, TensorIndex];

/**
 * A Keras JSON entry representing a Node, i.e. a specific instance of a Layer.
 *
 * By Keras JSON convention, a Node is specified as an array of Tensor keys
 * (i.e., references to Tensors output by other Layers) providing the inputs to
 * this Layer in order.
 */
export type NodeConfig = TensorKeyWithArgsArray[];

/**
 * A Keras JSON entry representing a layer.
 *
 * This is a 'Wrapper' because the Keras JSON convention is to provide the
 * `className` (i.e., the layer type) at the top level, and then to place the
 * layer-specific configuration in a `config` subtree.  These layer-specific
 * configurations are provided by subtypes of `LayerConfig` from tfjs-layers.
 * Thus, this 'wrapper' has a type paramater giving the specific type of the
 * wrapped `LayerConfig`.
 */
export interface LayerWrapper<T extends LayerConfig> {
  className: string;
  name: string;
  inboundNodes?: NodeConfig[];
  config: T;
}

/**
 * A standard Keras JSON 'Model' configuration.
 */
export interface ModelWrapper {
  className: 'Model';
  config: {
    name: string,
    layers: Array<LayerWrapper<{}>>,
    inputLayers: TensorKeyArray[],
    outputLayers: TensorKeyArray[],
  };
}

/**
 * A standard Keras JSON 'Sequential' configuration.
 */
export interface SequentialWrapper {
  className: 'Sequential';
  config: {layers: Array<LayerWrapper<{}>>};
}

/**
 * A legacy Keras JSON 'Sequential' configuration.
 *
 * It was a bug that Keras Sequential models were recorded with
 * model_config.config as an array of layers, instead of a dict containing a
 * 'layers' entry.  While the bug has been fixed, we still need to be able to
 * read this legacy format.
 */
export interface LegacySequentialWrapper {
  className: 'Sequential';
  config: Array<LayerWrapper<{}>>;
}

export interface RegularizerWrapper<T extends RegularizerConfig> {
  className: string;
  config: T;
}
