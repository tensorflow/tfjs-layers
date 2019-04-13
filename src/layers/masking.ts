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
 * TensorFlow.js Layers: Masking Layer.
 */


import {serialization, Tensor, tidy, notEqual, any} from '@tensorflow/tfjs-core';

import {Layer, LayerArgs} from '../engine/topology';
import {Shape} from '../keras_format/common';
import {Kwargs} from '../types';
import {getExactlyOneTensor} from '../utils/types_utils';

export declare interface MaskingArgs extends LayerArgs {
 /** Mask value.  */
 mask_value: number;
}

/**
* Masks a sequence by using a mask value to skip timesteps.
*
* If all features for a given sample timestep are equal to `mask_value`,
* then the sample timestep will be masked (skipped) in all downstream layers
* (as long as they support masking).
*
* If any downstream layer does not support masking yet receives such
* an input mask, an exception will be raised.
*
* # Arguments
*     mask_value: Either None or mask value to skip.
*
* # Input shape
*     Arbitrary. Use the keyword argument `input_shape`
*     (tuple of integers, does not include the samples axis)
*     when using this layer as the first layer in a model.
*
* # Output shape
*     Same shape as input.
*/
export class Masking extends Layer {

  static className = 'Masking';
  readonly mask_value: number;

  constructor(args: MaskingArgs) {
   super(args);
   this.supportsMasking = true;
   this.mask_value = args.mask_value;
 }

  computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
   return inputShape;
 }

  getConfig() {
   const baseConfig = super.getConfig();
   const config = {mask_value: this.mask_value};
   Object.assign(config, baseConfig);
   return config;
 }

  call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[] {
   return tidy(() => {
     this.invokeCallHook(inputs, kwargs);
     const input = getExactlyOneTensor(inputs);
     const boolean_mask = any(notEqual(input,this.mask_value),-1,true);
     const output = input.mul(boolean_mask.asType(input.dtype));
     return output;
   });
 }
}
serialization.registerClass(Masking);
