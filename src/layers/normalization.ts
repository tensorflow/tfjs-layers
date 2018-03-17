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
 * Normalization layers.
 */

import {Tensor} from '@tensorflow/tfjs-core';
import * as _ from 'underscore';

import * as K from '../backend/deeplearnjs_backend';
import * as constraints from '../constraints';
import {InputSpec, Layer, LayerConfig} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import * as initializers from '../initializers';
import * as regularizers from '../regularizers';
import {Shape} from '../types';
import {ConfigDict, LayerVariable} from '../types';
import * as generic_utils from '../utils/generic_utils';

export interface BatchNormalizationLayerConfig extends LayerConfig {
  /**
   * Integer, the axis that should be normalized (typically the features axis).
   * For instance, after a `Conv2D` layer with `data_format="channels_first"`,
   * set `axis=1` in `BatchNormalization`.
   * Default: -1.
   */
  axis?: number;

  /**
   * Momentum of the moving average.
   * Default: 0.99.
   */
  momentum?: number;

  /**
   * Small float added to the variance to avoid dividing by zero.
   * Default: 1e-3.
   */
  epsilon?: number;

  /**
   * If `true`, add offset of `beta` to normalized tensor.
   * If `false`, `beta` is ignored.
   * Default: true.
   */
  center?: boolean;

  /**
   * If `true`, multiply by `gamma`.
   * If `false`, `gamma` is not used.
   * When the next layer is linear (also e.g. `nn.relu`),
   * this can be disabled since the scaling will be done by the next layer.
   * Default: true.
   */
  scale?: boolean;

  /**
   * Initializer for the beta weight.
   * Default: 'Zeros'.
   */
  betaInitializer?: string|initializers.Initializer;

  /**
   * Initializer for the gamma weight.
   * Default: 'Ones'.
   */
  gammaInitializer?: string|initializers.Initializer;

  /**
   * Initializer for the moving mean.
   * Default: 'Zeros'
   */
  movingMeanInitializer?: string|initializers.Initializer;

  /**
   * Initializer for the moving variance.
   * Default: 'Ones'.
   */
  movingVarianceInitializer?: string|initializers.Initializer;

  /**
   * Optional constraint for the beta weight.
   */
  betaConstraint?: string|constraints.Constraint;

  /**
   * Optional constraint for gamma weight.
   */
  gammaConstraint?: string|constraints.Constraint;

  /**
   * Optional regularizer for the beta weight.
   */
  betaRegularizer?: string|regularizers.Regularizer;

  /**
   * Optional regularizer for the gamma weight.
   */
  gammaRegularizer?: string|regularizers.Regularizer;
}


/**
 * Batch normalization layer (Ioffe and Szegedy, 2014).
 *
 * Normalize the activations of the previous layer at each batch,
 * i.e. applies a transformation that maintains the mean activation
 * close to 0 and the activation standard deviation close to 1.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape` (Array of integers, does
 *   not include the sample axis) when calling the constructor of this class,
 *   if this layer is used as a first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Batch Normalization: Accelerating Deep Network Training by Reducing
 * Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
 */
export class BatchNormalization extends Layer {
  private readonly axis: number;
  private readonly momentum: number;
  private readonly epsilon: number;
  private readonly center: boolean;
  private readonly scale: boolean;
  private readonly betaInitializer: initializers.Initializer;
  private readonly gammaInitializer: initializers.Initializer;
  private readonly movingMeanInitializer: initializers.Initializer;
  private readonly movingVarianceInitializer: initializers.Initializer;
  private readonly betaConstraint: constraints.Constraint;
  private readonly gammaConstraint: constraints.Constraint;
  private readonly betaRegularizer: regularizers.Regularizer;
  private readonly gammaRegularizer: regularizers.Regularizer;
  private gamma: LayerVariable;
  private beta: LayerVariable;
  private movingMean: LayerVariable;
  private movingVariance: LayerVariable;

  constructor(config: BatchNormalizationLayerConfig) {
    super(config);
    this.supportsMasking = true;
    this.axis = config.axis == null ? -1 : config.axis;
    this.momentum = config.momentum == null ? 0.99 : config.momentum;
    this.epsilon = config.epsilon == null ? 1e-3 : config.epsilon;
    this.center = config.center == null ? true : config.center;
    this.scale = config.scale == null ? true : config.scale;
    this.betaInitializer =
        initializers.getInitializer(config.betaInitializer || 'Zeros');
    this.gammaInitializer =
        initializers.getInitializer(config.gammaInitializer || 'Ones');
    this.movingMeanInitializer =
        initializers.getInitializer(config.movingMeanInitializer || 'Zeros');
    this.movingVarianceInitializer =
        initializers.getInitializer(config.movingVarianceInitializer || 'Ones');
    this.betaConstraint = constraints.getConstraint(config.betaConstraint);
    this.gammaConstraint = constraints.getConstraint(config.gammaConstraint);
    this.betaRegularizer = regularizers.getRegularizer(config.betaRegularizer);
    this.gammaRegularizer =
        regularizers.getRegularizer(config.gammaRegularizer);
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = generic_utils.getExactlyOneShape(inputShape);
    const axis = this.axis >= 0 ? this.axis : (this.axis + inputShape.length);
    const dim = inputShape[axis];
    if (dim == null) {
      throw new ValueError(
          `Axis ${axis} of input tensor should have a defined dimension but ` +
          `the layer received an input with shape ` +
          `${JSON.stringify(inputShape)}.`);
    }
    this.inputSpec =
        [new InputSpec({ndim: inputShape.length, axes: {[axis]: dim}})];
    const shape = [dim];
    if (this.scale) {
      this.gamma = this.addWeight(
          'gamma', shape, null, this.gammaInitializer, this.gammaRegularizer,
          true, this.gammaConstraint);
    }
    if (this.center) {
      this.beta = this.addWeight(
          'beta', shape, null, this.betaInitializer, this.betaRegularizer, true,
          this.betaConstraint);
    }
    this.movingMean = this.addWeight(
        'moving_mean', shape, null, this.movingMeanInitializer, null, false);
    this.movingVariance = this.addWeight(
        'moving_variance', shape, null, this.movingVarianceInitializer, null,
        false);
    this.built = true;
  }

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    const training = kwargs['training'] == null ? false : kwargs['training'];
    const input = generic_utils.getExactlyOneTensor(inputs);
    const inputShape = K.shape(input);
    const ndim = inputShape.length;
    const reductionAxes = _.range(ndim);
    const axis = this.axis >= 0 ? this.axis : (this.axis + ndim);
    reductionAxes.splice(axis, 1);
    const broadcastShape = generic_utils.pyListRepeat(1, ndim);
    broadcastShape[axis] = inputShape[axis];

    const sortedReductionAxes = reductionAxes.slice();
    sortedReductionAxes.sort();
    const needsBroadcasting =
        !_.isEqual(sortedReductionAxes, _.range(ndim).slice(0, ndim - 1));

    const normalizeInference: () => Tensor = () => {
      if (needsBroadcasting) {
        const broadcastMovingMean =
            K.reshape(this.movingMean.read(), broadcastShape);
        const broadcastMovingVariance =
            K.reshape(this.movingVariance.read(), broadcastShape);
        const broadcastBeta =
            this.center ? K.reshape(this.beta.read(), broadcastShape) : null;
        const broadcastGamma =
            this.center ? K.reshape(this.gamma.read(), broadcastShape) : null;
        return K.batchNormalization(
            input, broadcastMovingMean, broadcastMovingVariance, broadcastBeta,
            broadcastGamma, this.epsilon);
      } else {
        return K.batchNormalization(
            input, this.movingMean.read(), this.movingVariance.read(),
            this.beta.read(), this.gamma.read(), this.epsilon);
      }
    };

    if (!training) {
      return normalizeInference();
    }

    throw new NotImplementedError(
        'BatchNormalization.call() has not been implemented for training ' +
        'mode yet.');
  }

  getConfig(): ConfigDict {
    const config: ConfigDict = {
      axis: this.axis,
      momentum: this.momentum,
      epsilon: this.epsilon,
      center: this.center,
      scale: this.scale,
      betaInitializer: initializers.serializeInitializer(this.betaInitializer),
      gammaInitializer:
          initializers.serializeInitializer(this.gammaInitializer),
      movingMeanInitializer:
          initializers.serializeInitializer(this.movingMeanInitializer),
      movingVarianceInitializer:
          initializers.serializeInitializer(this.movingVarianceInitializer),
      betaRegularizer: regularizers.serializeRegularizer(this.betaRegularizer),
      gammaRegularizer:
          regularizers.serializeRegularizer(this.gammaRegularizer),
      betaConstraint: constraints.serializeConstraint(this.betaConstraint),
      gammaConstraint: constraints.serializeConstraint(this.gammaConstraint)
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
generic_utils.ClassNameMap.register('BatchNormalization', BatchNormalization);
