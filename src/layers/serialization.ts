/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original Source layers/__init__.py */
import {serialization} from '@tensorflow/tfjs-core';

import {deserializeKerasObject} from '../utils/generic_utils';

/**
 * Instantiate a layer from a config dictionary.
 * @param config: dict of the form {class_name: str, config: dict}
 * @param custom_objects: dict mapping class names (or function names)
 *      of custom (non-Keras) objects to class/functions
 * @returns Layer instance (may be Model, Sequential, Layer...)
 */
export function deserialize(
    config: serialization.ConfigDict,
    customObjects = {} as
        serialization.ConfigDict): serialization.Serializable {
  return deserializeKerasObject(
      config, serialization.SerializationMap.getMap().classNameMap,
      customObjects, 'layer');
}
