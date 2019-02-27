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
 * TensorFlow.js Layers: Basic Layers.
 */

import {Scalar, serialization, Tensor, tidy, transpose, util} from '@tensorflow/tfjs-core';

import {Activation as ActivationFn, getActivation, serializeActivation} from '../activations';
import {getScalar} from '../backend/state';
import * as K from '../backend/tfjs_backend';
import {Constraint, ConstraintIdentifier, getConstraint, serializeConstraint} from '../constraints';
import {DisposeResult, InputSpec, Layer, LayerArgs} from '../engine/topology';
import {NotImplementedError, ValueError} from '../errors';
import {getInitializer, Initializer, InitializerIdentifier, serializeInitializer} from '../initializers';
import {ActivationIdentifier} from '../keras_format/activation_config';
import {Shape} from '../keras_format/common';
import {getRegularizer, Regularizer, RegularizerIdentifier, serializeRegularizer} from '../regularizers';
import {Kwargs} from '../types';
import {assertPositiveInteger} from '../utils/generic_utils';
import {arrayProd, range} from '../utils/math_utils';
import {getExactlyOneShape, getExactlyOneTensor} from '../utils/types_utils';
import {LayerVariable} from '../variables';

export declare interface DropoutLayerArgs extends LayerArgs {
  /** Float between 0 and 1. Fraction of the input units to drop. */
  rate: number;

  /**
   * Integer array representing the shape of the binary dropout mask that will
   * be multiplied with the input.
   *
   * For instance, if your inputs have shape `(batchSize, timesteps, features)`
   * and you want the dropout mask to be the same for all timesteps, you can use
   * `noise_shape=(batch_size, 1, features)`.
   */
  noiseShape?: number[];

  /** An integer to use as random seed. */
  seed?: number;
}

/**
 * Applies
 * [dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) to
 * the input.
 *
 * Dropout consists in randomly setting a fraction `rate` of input units to 0 at
 * each update during training time, which helps prevent overfitting.
 */
export class Dropout extends Layer {
  /** @nocollapse */
  static className = 'Dropout';
  private readonly rate: number;
  private readonly rateScalar: Scalar;
  private readonly noiseShape: number[];
  private readonly seed: number;

  constructor(args: DropoutLayerArgs) {
    super(args);
    this.rate = Math.max(Math.min(args.rate, 1), 0);
    this.rateScalar = getScalar(this.rate);
    // So that the scalar doesn't get tidied up between executions.
    this.noiseShape = args.noiseShape;
    this.seed = args.seed;
    if (this.seed != null) {
      throw new NotImplementedError(
          'Non-default seed is not implemented in Dropout layer yet: ' +
          this.seed);
    }
    this.supportsMasking = true;
  }

  private getNoiseShape(input: Tensor): Shape {
    if (this.noiseShape == null) {
      return this.noiseShape;
    }
    const inputShape = input.shape;
    const noiseShape: Shape = [];
    for (let i = 0; i < this.noiseShape.length; ++i) {
      noiseShape.push(
          this.noiseShape[i] == null ? inputShape[i] : this.noiseShape[i]);
    }
    return noiseShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      if (this.noiseShape != null &&
          !util.arraysEqual(input.shape, this.noiseShape)) {
        throw new NotImplementedError(
            'Non-default noise shape is not implemented in Dropout ' +
            'layer yet: ' + JSON.stringify(this.noiseShape));
      }
      if (0 < this.rate && this.rate < 1) {
        const training =
            kwargs['training'] == null ? false : kwargs['training'];
        const noiseShape = this.getNoiseShape(input);
        const output =
            K.inTrainPhase(
                () => K.dropout(input, this.rateScalar, noiseShape, this.seed),
                () => input, training) as Tensor;
        return output;
      }
      return inputs;
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      rate: this.rate,
      noiseShape: this.noiseShape,
      seed: this.seed,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  dispose(): DisposeResult {
    const result = super.dispose();
    if (!this.rateScalar.isDisposed) {
      this.rateScalar.dispose();
      result.numDisposedVariables++;
    }
    return result;
  }
}
serialization.registerClass(Dropout);

export declare interface DenseLayerArgs extends LayerArgs {
  /** Positive integer, dimensionality of the output space. */
  units: number;
  /**
   * Activation function to use.
   *
   * If unspecified, no activation is applied.
   */
  activation?: ActivationIdentifier;
  /** Whether to apply a bias. */
  useBias?: boolean;
  /**
   * Initializer for the dense kernel weights matrix.
   */
  kernelInitializer?: InitializerIdentifier|Initializer;
  /**
   * Initializer for the bias vector.
   */
  biasInitializer?: InitializerIdentifier|Initializer;
  /**
   * If specified, defines inputShape as `[inputDim]`.
   */
  inputDim?: number;

  /**
   * Constraint for the kernel weights.
   */
  kernelConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Constraint for the bias vector.
   */
  biasConstraint?: ConstraintIdentifier|Constraint;

  /**
   * Regularizer function applied to the dense kernel weights matrix.
   */
  kernelRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer function applied to the bias vector.
   */
  biasRegularizer?: RegularizerIdentifier|Regularizer;

  /**
   * Regularizer function applied to the activation.
   */
  activityRegularizer?: RegularizerIdentifier|Regularizer;
}

/**
 * Creates a dense (fully connected) layer.
 *
 * This layer implements the operation:
 *   `output = activation(dot(input, kernel) + bias)`
 *
 * `activation` is the element-wise activation function
 *   passed as the `activation` argument.
 *
 * `kernel` is a weights matrix created by the layer.
 *
 * `bias` is a bias vector created by the layer (only applicable if `useBias`
 * is `true`).
 *
 * **Input shape:**
 *
 *   nD `tf.Tensor` with shape: `(batchSize, ..., inputDim)`.
 *
 *   The most common situation would be
 *   a 2D input with shape `(batchSize, inputDim)`.
 *
 * **Output shape:**
 *
 *   nD tensor with shape: `(batchSize, ..., units)`.
 *
 *   For instance, for a 2D input with shape `(batchSize, inputDim)`,
 *   the output would have shape `(batchSize, units)`.
 *
 * Note: if the input to the layer has a rank greater than 2, then it is
 * flattened prior to the initial dot product with the kernel.
 */
export class Dense extends Layer {
  /** @nocollapse */
  static className = 'Dense';
  private units: number;
  // Default activation: Linear (none).
  private activation: ActivationFn = null;
  private useBias = true;
  private kernelInitializer: Initializer;
  private biasInitializer: Initializer;
  private kernel: LayerVariable = null;
  private bias: LayerVariable = null;

  readonly DEFAULT_KERNEL_INITIALIZER: InitializerIdentifier = 'glorotNormal';
  readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier = 'zeros';
  private readonly kernelConstraint?: Constraint;
  private readonly biasConstraint?: Constraint;
  private readonly kernelRegularizer?: Regularizer;
  private readonly biasRegularizer?: Regularizer;

  constructor(args: DenseLayerArgs) {
    super(args);
    if (args.batchInputShape == null && args.inputShape == null &&
        args.inputDim != null) {
      // This logic is copied from Layer's constructor, since we can't
      // do exactly what the Python constructor does for Dense().
      let batchSize: number = null;
      if (args.batchSize != null) {
        batchSize = args.batchSize;
      }
      this.batchInputShape = [batchSize, args.inputDim];
    }

    this.units = args.units;
    assertPositiveInteger(this.units, 'units');
    this.activation = getActivation(args.activation);
    if (args.useBias != null) {
      this.useBias = args.useBias;
    }
    this.kernelInitializer = getInitializer(
        args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER);
    this.biasInitializer =
        getInitializer(args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER);
    this.kernelConstraint = getConstraint(args.kernelConstraint);
    this.biasConstraint = getConstraint(args.biasConstraint);
    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    this.biasRegularizer = getRegularizer(args.biasRegularizer);
    this.activityRegularizer = getRegularizer(args.activityRegularizer);
    this.supportsMasking = true;

    this.inputSpec = [{minNDim: 2}];
  }

  public build(inputShape: Shape|Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);
    const inputLastDim = inputShape[inputShape.length - 1];
    if (this.kernel == null) {
      this.kernel = this.addWeight(
          'kernel', [inputLastDim, this.units], null, this.kernelInitializer,
          this.kernelRegularizer, true, this.kernelConstraint);
      if (this.useBias) {
        this.bias = this.addWeight(
            'bias', [this.units], null, this.biasInitializer,
            this.biasRegularizer, true, this.biasConstraint);
      }
    }

    this.inputSpec = [{minNDim: 2, axes: {[-1]: inputLastDim}}];
    this.built = true;
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const outputShape = inputShape.slice();
    outputShape[outputShape.length - 1] = this.units;
    return outputShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      // Dense layer accepts only a single input.
      const input = getExactlyOneTensor(inputs);
      let output = K.dot(input, this.kernel.read());
      if (this.bias != null) {
        output = K.biasAdd(output, this.bias.read());
      }
      if (this.activation != null) {
        output = this.activation.apply(output);
      }
      return output;
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint)
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Dense);

/**
 * Flattens the input. Does not affect the batch size.
 *
 * A `Flatten` layer flattens each batch in its inputs to 1D (making the output
 * 2D).
 *
 * For example:
 *
 * ```js
 * const input = tf.input({shape: [4, 3]});
 * const flattenLayer = tf.layers.flatten();
 * // Inspect the inferred output shape of the flatten layer, which
 * // equals `[null, 12]`. The 2nd dimension is 4 * 3, i.e., the result of the
 * // flattening. (The 1st dimension is the undermined batch size.)
 * console.log(JSON.stringify(flattenLayer.apply(input).shape));
 * ```
 */
export class Flatten extends Layer {
  /** @nocollapse */
  static className = 'Flatten';
  constructor(args?: LayerArgs) {
    super(args || {});
    this.inputSpec = [{minNDim: 3}];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    for (const dim of inputShape.slice(1)) {
      if (dim == null) {
        throw new ValueError(
            `The shape of the input to "Flatten" is not fully defined ` +
            `(got ${inputShape.slice(1)}). Make sure to pass a complete ` +
            `"input_shape" or "batch_input_shape" argument to the first ` +
            `layer in your model.`);
      }
    }
    return [inputShape[0], arrayProd(inputShape, 1)];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      return K.batchFlatten(getExactlyOneTensor(inputs));
    });
  }
}
serialization.registerClass(Flatten);

export declare interface ActivationLayerArgs extends LayerArgs {
  /**
   * Name of the activation function to use.
   */
  activation: ActivationIdentifier;
}

/**
 * Applies an activation function to an output.
 *
 * This layer applies element-wise activation function.  Other layers, notably
 * `dense` can also apply activation functions.  Use this isolated activation
 * function to extract the values before and after the
 * activation. For instance:
 *
 * ```js
 * const input = tf.input({shape: [5]});
 * const denseLayer = tf.layers.dense({units: 1});
 * const activationLayer = tf.layers.activation({activation: 'relu6'});
 *
 * // Obtain the output symbolic tensors by applying the layers in order.
 * const denseOutput = denseLayer.apply(input);
 * const activationOutput = activationLayer.apply(denseOutput);
 *
 * // Create the model based on the inputs.
 * const model = tf.LayersModel({
 *     inputs: input,
 *     outputs: [denseOutput, activationOutput]
 * });
 *
 * // Collect both outputs and print separately.
 * const [denseOut, activationOut] = model.predict(tf.randomNormal([6, 5]));
 * denseOut.print();
 * activationOut.print();
 * ```
 *
 */
export class Activation extends Layer {
  /** @nocollapse */
  static className = 'Activation';
  activation: ActivationFn;

  constructor(args: ActivationLayerArgs) {
    super(args);
    this.supportsMasking = true;
    this.activation = getActivation(args.activation);
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      return this.activation.apply(input);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {activation: serializeActivation(this.activation)};
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Activation);

export declare interface ReshapeLayerArgs extends LayerArgs {
  /** The target shape. Does not include the batch axis. */
  targetShape: Shape;
}

export declare interface RepeatVectorLayerArgs extends LayerArgs {
  /**
   * The integer number of times to repeat the input.
   */
  n: number;
}

/**
 * Repeats the input n times in a new dimension.
 *
 * ```js
 *  const model = tf.sequential();
 *  model.add(tf.layers.repeatVector({n: 4, inputShape: [2]}));
 *  const x = tf.tensor2d([[10, 20]]);
 *  // Use the model to do inference on a data point the model hasn't see
 *  model.predict(x).print();
 *  // output shape is now [batch, 2, 4]
 * ```
 */
export class RepeatVector extends Layer {
  /** @nocollapse */
  static className = 'RepeatVector';
  readonly n: number;

  constructor(args: RepeatVectorLayerArgs) {
    super(args);
    this.n = args.n;
    this.inputSpec = [{ndim: 2}];
  }

  computeOutputShape(inputShape: Shape): Shape {
    return [inputShape[0], this.n, inputShape[1]];
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      inputs = getExactlyOneTensor(inputs);
      return K.repeat(inputs, this.n);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      n: this.n,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(RepeatVector);

/**
 * Reshapes an input to a certain shape.
 *
 * ```js
 * const input = tf.input({shape: [4, 3]});
 * const reshapeLayer = tf.layers.reshape({targetShape: [2, 6]});
 * // Inspect the inferred output shape of the Reshape layer, which
 * // equals `[null, 2, 6]`. (The 1st dimension is the undermined batch size.)
 * console.log(JSON.stringify(reshapeLayer.apply(input).shape));
 * ```
 *
 * Input shape:
 *   Arbitrary: although all dimensions in the input shape must be fixed.
 *     Use the ReshapeLayerConfig field `input_shape` when using this layer
 *     as the first layer in a model.
 *
 * Output shape:
 *   [batchSize, targetShape[0], targetShape[1], ...,
 *    targetShape[targetShape.length - 1]].
 */
export class Reshape extends Layer {
  /** @nocollapse */
  static className = 'Reshape';
  private targetShape: Shape;

  constructor(args: ReshapeLayerArgs) {
    super(args);
    this.targetShape = args.targetShape;

    // Make sure that all unknown dimensions are represented as `null`.
    for (let i = 0; i < this.targetShape.length; ++i) {
      if (this.isUnknown(this.targetShape[i])) {
        this.targetShape[i] = null;
      }
    }
  }

  private isUnknown(dim: number): boolean {
    return dim < 0 || dim == null;
  }

  /**
   * Finds and replaces a missing dimension in output shape.
   *
   * This is a near direct port of the internal Numpy function
   * `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`.
   *
   * @param inputShape: Original shape of array begin reshape.
   * @param outputShape: Target shape of the array, with at most a single
   * `null` or negative number, which indicates an underdetermined dimension
   * that should be derived from `inputShape` and the known dimensions of
   *   `outputShape`.
   * @returns: The output shape with `null` replaced with its computed value.
   * @throws: ValueError: If `inputShape` and `outputShape` do not match.
   */
  private fixUnknownDimension(inputShape: Shape, outputShape: Shape): Shape {
    const errorMsg = 'Total size of new array must be unchanged.';
    const finalShape = outputShape.slice();
    let known = 1;
    let unknown = null;
    for (let i = 0; i < finalShape.length; ++i) {
      const dim = finalShape[i];
      if (this.isUnknown(dim)) {
        if (unknown === null) {
          unknown = i;
        } else {
          throw new ValueError('Can only specifiy one unknown dimension.');
        }
      } else {
        known *= dim;
      }
    }

    const originalSize = arrayProd(inputShape);
    if (unknown !== null) {
      if (known === 0 || originalSize % known !== 0) {
        throw new ValueError(errorMsg);
      }
      finalShape[unknown] = originalSize / known;
    } else if (originalSize !== known) {
      throw new ValueError(errorMsg);
    }

    return finalShape;
  }

  computeOutputShape(inputShape: Shape): Shape {
    let anyUnknownDims = false;
    for (let i = 0; i < inputShape.length; ++i) {
      if (this.isUnknown(inputShape[i])) {
        anyUnknownDims = true;
        break;
      }
    }

    if (anyUnknownDims) {
      return inputShape.slice(0, 1).concat(this.targetShape);
    } else {
      return inputShape.slice(0, 1).concat(
          this.fixUnknownDimension(inputShape.slice(1), this.targetShape));
    }
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = getExactlyOneTensor(inputs);
      const inputShape = input.shape;
      const outputShape = inputShape.slice(0, 1).concat(
          this.fixUnknownDimension(inputShape.slice(1), this.targetShape));
      return input.reshape(outputShape);
    });
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      targetShape: this.targetShape,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Reshape);

export declare interface PermuteLayerArgs extends LayerArgs {
  /**
   * Array of integers. Permutation pattern. Does not include the
   * sample (batch) dimension. Index starts at 1.
   * For instance, `[2, 1]` permutes the first and second dimensions
   * of the input.
   */
  dims: number[];
}

/**
 * Permutes the dimensions of the input according to a given pattern.
 *
 * Useful for, e.g., connecting RNNs and convnets together.
 *
 * Example:
 *
 * ```js
 * const model = tf.Sequential();
 * model.add(tf.layers.permute({
 *   dims: [2, 1],
 *   inputShape: [10, 64]
 * }));
 * console.log(model.outputShape);
 * // Now model's output shape is [null, 64, 10], where null is the
 * // unpermuted sample (batch) dimension.
 * ```
 *
 * Input shape:
 *   Arbitrary. Use the configuration field `inputShape` when using this
 *   layer as othe first layer in a model.
 *
 * Output shape:
 *   Same rank as the input shape, but with the dimensions re-ordered (i.e.,
 *   permuted) according to the `dims` configuration of this layer.
 */
export class Permute extends Layer {
  /** @nocollapse */
  static className = 'Permute';
  readonly dims: number[];
  private readonly dimsIncludingBatch: number[];

  constructor(args: PermuteLayerArgs) {
    super(args);
    if (args.dims == null) {
      throw new Error(
          'Required configuration field `dims` is missing during Permute ' +
          'constructor call.');
    }
    if (!Array.isArray(args.dims)) {
      throw new Error(
          'Permute constructor requires `dims` to be an Array, but received ' +
          `${args.dims} instead.`);
    }

    // Check the validity of the permutation indices.
    const expectedSortedIndices = range(1, args.dims.length + 1);
    if (!util.arraysEqual(args.dims.slice().sort(), expectedSortedIndices)) {
      throw new Error(
          'Invalid permutation `dims`: ' + JSON.stringify(args.dims) +
          ' `dims` must contain consecutive integers starting from 1.');
    }

    this.dims = args.dims;
    this.dimsIncludingBatch = [0].concat(this.dims);
    this.inputSpec = [new InputSpec({ndim: this.dims.length + 1})];
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const outputShape = inputShape.slice();
    this.dims.forEach((dim: number, i: number) => {
      outputShape[i + 1] = (inputShape as Shape)[dim];
    });
    return outputShape;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return transpose(getExactlyOneTensor(inputs), this.dimsIncludingBatch);
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      dims: this.dims,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(Permute);
