/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {DataType, eye, linalg, mul, ones, randomUniform, scalar, Scalar, serialization, Tensor, Tensor2D, tidy, truncatedNormal, zeros} from '@tensorflow/tfjs-core';

import {getScalar} from './backend/state';
import * as K from './backend/tfjs_backend';
import {checkDataFormat, DataFormat} from './common';
import {NotImplementedError, ValueError} from './errors';
import {checkDistribution, checkFanMode, ConstantBaseConfig, Distribution, FanMode, IdentityBaseConfig, InitializerIdentifier, OrthogonalBaseConfig, RandomNormalBaseConfig, RandomUniformBaseConfig, SeedOnlyInitializerBaseConfig, TruncatedNormalBaseConfig, VarianceScalingBaseConfig} from './keras_format/initializer_config';
import {Shape} from './keras_format/types';
import {deserializeKerasObject, serializeKerasObject} from './utils/generic_utils';
import {arrayProd} from './utils/math_utils';


/**
 * Initializer base class.
 *
 * @doc {
 *   heading: 'Initializers', subheading: 'Classes', namespace: 'initializers'}
 */
export abstract class Initializer extends serialization.Serializable {
  public fromConfigUsesCustomObjects(): boolean {
    return false;
  }
  /**
   * Generate an initial value.
   * @param shape
   * @param dtype
   * @return The init value.
   */
  abstract apply(shape: Shape, dtype?: DataType): Tensor;

  getConfig(): serialization.ConfigDict {
    return {};
  }
}

/**
 * Initializer that generates tensors initialized to 0.
 */
export class Zeros extends Initializer {
  static className = 'Zeros';

  apply(shape: Shape, dtype?: DataType): Tensor {
    return zeros(shape, dtype);
  }
}
serialization.registerClass(Zeros);

/**
 * Initializer that generates tensors initialized to 1.
 */
export class Ones extends Initializer {
  static className = 'Ones';

  apply(shape: Shape, dtype?: DataType): Tensor {
    return ones(shape, dtype);
  }
}
serialization.registerClass(Ones);

export type ConstantArgs = ConstantBaseConfig;

/**
 * Initializer that generates values initialized to some constant.
 */
export class Constant extends Initializer {
  static className = 'Constant';
  private value: number;
  constructor(args: ConstantArgs) {
    super();
    if (typeof args !== 'object') {
      throw new ValueError(
          `Expected argument of type ConstantConfig but got ${args}`);
    }
    if (args.value === undefined) {
      throw new ValueError(`config must have value set but got ${args}`);
    }
    this.value = args.value;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    return tidy(() => mul(scalar(this.value), ones(shape, dtype)));
  }

  getConfig(): serialization.ConfigDict {
    return {
      value: this.value,
    };
  }
}
serialization.registerClass(Constant);

export type RandomUniformArgs = RandomUniformBaseConfig;

/**
 * Initializer that generates random values initialized to a uniform
 * distribution.
 *
 * Values will be distributed uniformly between the configured minval and
 * maxval.
 */
export class RandomUniform extends Initializer {
  static className = 'RandomUniform';
  readonly DEFAULT_MINVAL = -0.05;
  readonly DEFAULT_MAXVAL = 0.05;
  private minval: number;
  private maxval: number;
  private seed: number;

  constructor(args: RandomUniformArgs) {
    super();
    this.minval = args.minval || this.DEFAULT_MINVAL;
    this.maxval = args.maxval || this.DEFAULT_MAXVAL;
    this.seed = args.seed;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    return randomUniform(shape, this.minval, this.maxval, dtype);
  }

  getConfig(): serialization.ConfigDict {
    return {minval: this.minval, maxval: this.maxval, seed: this.seed};
  }
}
serialization.registerClass(RandomUniform);

export type RandomNormalArgs = RandomNormalBaseConfig;

/**
 * Initializer that generates random values initialized to a normal
 * distribution.
 */
export class RandomNormal extends Initializer {
  static className = 'RandomNormal';
  readonly DEFAULT_MEAN = 0.;
  readonly DEFAULT_STDDEV = 0.05;
  private mean: number;
  private stddev: number;
  private seed: number;

  constructor(args: RandomNormalArgs) {
    super();
    this.mean = args.mean || this.DEFAULT_MEAN;
    this.stddev = args.stddev || this.DEFAULT_STDDEV;
    this.seed = args.seed;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    dtype = dtype || 'float32';
    if (dtype !== 'float32' && dtype !== 'int32') {
      throw new NotImplementedError(
          `randomNormal does not support dType ${dtype}.`);
    }

    return K.randomNormal(shape, this.mean, this.stddev, dtype, this.seed);
  }

  getConfig(): serialization.ConfigDict {
    return {mean: this.mean, stddev: this.stddev, seed: this.seed};
  }
}
serialization.registerClass(RandomNormal);

export type TruncatedNormalArgs = TruncatedNormalBaseConfig;

/**
 * Initializer that generates random values initialized to a truncated normal.
 * distribution.
 *
 * These values are similar to values from a `RandomNormal` except that values
 * more than two standard deviations from the mean are discarded and re-drawn.
 * This is the recommended initializer for neural network weights and filters.
 */
export class TruncatedNormal extends Initializer {
  static className = 'TruncatedNormal';

  readonly DEFAULT_MEAN = 0.;
  readonly DEFAULT_STDDEV = 0.05;
  private mean: number;
  private stddev: number;
  private seed: number;

  constructor(args: TruncatedNormalArgs) {
    super();
    this.mean = args.mean || this.DEFAULT_MEAN;
    this.stddev = args.stddev || this.DEFAULT_STDDEV;
    this.seed = args.seed;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    dtype = dtype || 'float32';
    if (dtype !== 'float32' && dtype !== 'int32') {
      throw new NotImplementedError(
          `truncatedNormal does not support dType ${dtype}.`);
    }
    return truncatedNormal(shape, this.mean, this.stddev, dtype, this.seed);
  }

  getConfig(): serialization.ConfigDict {
    return {mean: this.mean, stddev: this.stddev, seed: this.seed};
  }
}
serialization.registerClass(TruncatedNormal);

export type IdentityArgs = IdentityBaseConfig;

/**
 * Initializer that generates the identity matrix.
 * Only use for square 2D matrices.
 */
export class Identity extends Initializer {
  static className = 'Identity';
  private gain: Scalar;
  constructor(args: IdentityArgs) {
    super();
    this.gain = args.gain != null ? scalar(args.gain) : getScalar(1.0);
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    return tidy(() => {
      if (shape.length !== 2 || shape[0] !== shape[1]) {
        throw new ValueError(
            'Identity matrix initializer can only be used for' +
            ' 2D square matrices.');
      } else {
        return mul(this.gain, eye(shape[0]));
      }
    });
  }

  getConfig(): serialization.ConfigDict {
    return {gain: this.gain.get()};
  }
}
serialization.registerClass(Identity);

/**
 * Computes the number of input and output units for a weight shape.
 * @param shape Shape of weight.
 * @param dataFormat data format to use for convolution kernels.
 *   Note that all kernels in Keras are standardized on the
 *   CHANNEL_LAST ordering (even when inputs are set to CHANNEL_FIRST).
 * @return An length-2 array: fanIn, fanOut.
 */
function computeFans(
    shape: Shape, dataFormat: DataFormat = 'channelsLast'): number[] {
  let fanIn: number;
  let fanOut: number;
  checkDataFormat(dataFormat);
  if (shape.length === 2) {
    fanIn = shape[0];
    fanOut = shape[1];
  } else if ([3, 4, 5].indexOf(shape.length) !== -1) {
    if (dataFormat === 'channelsFirst') {
      const receptiveFieldSize = arrayProd(shape, 2);
      fanIn = shape[1] * receptiveFieldSize;
      fanOut = shape[0] * receptiveFieldSize;
    } else if (dataFormat === 'channelsLast') {
      const receptiveFieldSize = arrayProd(shape, 0, shape.length - 2);
      fanIn = shape[shape.length - 2] * receptiveFieldSize;
      fanOut = shape[shape.length - 1] * receptiveFieldSize;
    }
  } else {
    const shapeProd = arrayProd(shape);
    fanIn = Math.sqrt(shapeProd);
    fanOut = Math.sqrt(shapeProd);
  }

  return [fanIn, fanOut];
}

export type VarianceScalingArgs = VarianceScalingBaseConfig;

/**
 * Initializer capable of adapting its scale to the shape of weights.
 * With distribution=NORMAL, samples are drawn from a truncated normal
 * distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:
 *   - number of input units in the weight tensor, if mode = FAN_IN.
 *   - number of output units, if mode = FAN_OUT.
 *   - average of the numbers of input and output units, if mode = FAN_AVG.
 * With distribution=UNIFORM,
 * samples are drawn from a uniform distribution
 * within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
 */
export class VarianceScaling extends Initializer {
  static className = 'VarianceScaling';
  private scale: number;
  private mode: FanMode;
  private distribution: Distribution;
  private seed: number;

  /**
   * Constructor of VarianceScaling.
   * @throws ValueError for invalid value in scale.
   */
  constructor(args: VarianceScalingArgs) {
    super();
    if (args.scale < 0.0) {
      throw new ValueError(
          `scale must be a positive float. Got: ${args.scale}`);
    }
    this.scale = args.scale == null ? 1.0 : args.scale;
    this.mode = args.mode;
    checkFanMode(this.mode);
    this.distribution = args.distribution;
    checkDistribution(this.distribution);
    this.seed = args.seed;
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    const fans = computeFans(shape);
    const fanIn = fans[0];
    const fanOut = fans[1];
    let scale = this.scale;
    if (this.mode === 'fanIn') {
      scale /= Math.max(1, fanIn);
    } else if (this.mode === 'fanOut') {
      scale /= Math.max(1, fanOut);
    } else {
      scale /= Math.max(1, (fanIn + fanOut) / 2);
    }

    if (this.distribution === 'normal') {
      const stddev = Math.sqrt(scale);
      dtype = dtype || 'float32';
      if (dtype !== 'float32' && dtype !== 'int32') {
        throw new NotImplementedError(
            `${this.getClassName()} does not support dType ${dtype}.`);
      }
      return truncatedNormal(shape, 0, stddev, dtype, this.seed);
    } else {
      const limit = Math.sqrt(3 * scale);
      return randomUniform(shape, -limit, limit, dtype);
    }
  }

  getConfig(): serialization.ConfigDict {
    return {
      scale: this.scale,
      mode: this.mode,
      distribution: this.distribution,
      seed: this.seed
    };
  }
}
serialization.registerClass(VarianceScaling);

export type SeedOnlyInitializerArgs = SeedOnlyInitializerBaseConfig;

/**
 * Glorot uniform initializer, also called Xavier uniform initializer.
 * It draws samples from a uniform distribution within [-limit, limit]
 * where `limit` is `sqrt(6 / (fan_in + fan_out))`
 * where `fan_in` is the number of input units in the weight tensor
 * and `fan_out` is the number of output units in the weight tensor
 *
 * Reference:
 *   Glorot & Bengio, AISTATS 2010
 *       http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf.
 */
export class GlorotUniform extends VarianceScaling {
  static className = 'GlorotUniform';

  /**
   * Constructor of GlorotUniform
   * @param scale
   * @param mode
   * @param distribution
   * @param seed
   */
  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 1.0,
      mode: 'fanAvg',
      distribution: 'uniform',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, GlorotUniform is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(GlorotUniform);

/**
 * Glorot normal initializer, also called Xavier normal initializer.
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(2 / (fan_in + fan_out))`
 * where `fan_in` is the number of input units in the weight tensor
 * and `fan_out` is the number of output units in the weight tensor.
 *
 * Reference:
 *   Glorot & Bengio, AISTATS 2010
 *       http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
 */
export class GlorotNormal extends VarianceScaling {
  static className = 'GlorotNormal';

  /**
   * Constructor of GlorotNormal.
   * @param scale
   * @param mode
   * @param distribution
   * @param seed
   */
  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 1.0,
      mode: 'fanAvg',
      distribution: 'normal',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, GlorotNormal is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(GlorotNormal);

/**
 * He normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(2 / fanIn)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * Reference:
 *     He et al., http://arxiv.org/abs/1502.01852
 */
export class HeNormal extends VarianceScaling {
  static className = 'HeNormal';

  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 2.0,
      mode: 'fanIn',
      distribution: 'normal',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, HeNormal is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(HeNormal);

/**
 * LeCun normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(1 / fanIn)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * References:
 *   [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 *   [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
 */
export class LeCunNormal extends VarianceScaling {
  static className = 'LeCunNormal';

  constructor(args?: SeedOnlyInitializerArgs) {
    super({
      scale: 1.0,
      mode: 'fanIn',
      distribution: 'normal',
      seed: args == null ? null : args.seed
    });
  }

  getClassName(): string {
    // In Python Keras, LeCunNormal is not a class, but a helper method
    // that creates a VarianceScaling object. Use 'VarianceScaling' as
    // class name to be compatible with that.
    return VarianceScaling.className;
  }
}
serialization.registerClass(LeCunNormal);

export type OrthogonalArgs = OrthogonalBaseConfig;

/**
 * Initializer that generates a random orthogonal matrix.
 *
 * Reference:
 * [Saxe et al., http://arxiv.org/abs/1312.6120](http://arxiv.org/abs/1312.6120)
 */
export class Orthogonal extends Initializer {
  static className = 'Orthogonal';
  readonly DEFAULT_GAIN = 1;
  protected readonly gain: number;
  protected readonly seed: number;

  constructor(args?: OrthogonalArgs) {
    super();
    this.gain = args.gain == null ? this.DEFAULT_GAIN : args.gain;
    this.seed = args.seed;

    if (this.seed != null) {
      throw new NotImplementedError(
          'Random seed is not implemented for Orthogonal Initializer yet.');
    }
  }

  apply(shape: Shape, dtype?: DataType): Tensor {
    return tidy(() => {
      if (shape.length !== 2) {
        throw new NotImplementedError(
            'The Orthogonal Initializer does not support non-2D shapes yet.');
      }
      if (shape[0] * shape[1] > 2000) {
        console.warn(
            `Orthogonal initializer is being called on a matrix with more ` +
            `than 2000 (${shape[0] * shape[1]}) elements: ` +
            `Slowness may result.`);
      }

      // TODO(cais): Add seed support.
      const normalizedShape =
          shape[0] > shape[1] ? [shape[1], shape[0]] : shape;
      const a = K.randomNormal(normalizedShape, 0, 1, 'float32') as Tensor2D;
      let q = linalg.gramSchmidt(a) as Tensor2D;
      if (shape[0] > shape[1]) {
        q = q.transpose();
      }
      return mul(getScalar(this.gain), q);
    });
  }

  getConfig(): serialization.ConfigDict {
    return {
      gain: this.gain,
      seed: this.seed,
    };
  }
}
serialization.registerClass(Orthogonal);

// Maps the JavaScript-like identifier keys to the corresponding registry
// symbols.
export const INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP:
    {[identifier in InitializerIdentifier]: string} = {
      'constant': 'Constant',
      'glorotNormal': 'GlorotNormal',
      'glorotUniform': 'GlorotUniform',
      'heNormal': 'HeNormal',
      'identity': 'Identity',
      'leCunNormal': 'LeCunNormal',
      'ones': 'Ones',
      'orthogonal': 'Orthogonal',
      'randomNormal': 'RandomNormal',
      'randomUniform': 'RandomUniform',
      'truncatedNormal': 'TruncatedNormal',
      'varianceScaling': 'VarianceScaling',
      'zeros': 'Zeros'
    };

function deserializeInitializer(
    config: serialization.ConfigDict,
    customObjects: serialization.ConfigDict = {}): Initializer {
  return deserializeKerasObject(
      config, serialization.SerializationMap.getMap().classNameMap,
      customObjects, 'initializer');
}

export function serializeInitializer(initializer: Initializer):
    serialization.ConfigDictValue {
  return serializeKerasObject(initializer);
}

export function getInitializer(identifier: InitializerIdentifier|Initializer|
                               serialization.ConfigDict): Initializer {
  if (typeof identifier === 'string') {
    const className = identifier in INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
        INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
        identifier;
    /* We have four 'helper' classes for common initializers that
    all get serialized as 'VarianceScaling' and shouldn't go through
    the deserializeInitializer pathway. */
    if (className === 'GlorotUniform') {
      return new GlorotUniform();
    } else if (className === 'GlorotNormal') {
      return new GlorotNormal();
    } else if (className === 'HeNormal') {
      return new HeNormal();
    } else if (className === 'LeCunNormal') {
      return new LeCunNormal();
    } else {
      const config = {className, config: {}};
      return deserializeInitializer(config);
    }
  } else if (identifier instanceof Initializer) {
    return identifier;
  } else {
    return deserializeInitializer(identifier);
  }
}
