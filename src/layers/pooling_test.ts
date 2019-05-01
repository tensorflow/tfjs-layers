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
 * Unit tests for pooling.ts.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {Tensor, tensor2d, Tensor2D, tensor3d, tensor4d, Tensor4D, util} from '@tensorflow/tfjs-core';

import {SymbolicTensor} from '../engine/topology';
import * as tfl from '../index';
import {DataFormat, PaddingMode, PoolMode} from '../keras_format/common';
import {convOutputLength} from '../utils/conv_utils';
import {describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {pool2d} from './pooling';

describeMathCPUAndGPU('pool2d', () => {
  const x4by4Data = [[[
    [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
    [-20, -40, -60, -80]
  ]]];
  const x5by5Data = [[[
    [0, 1, 3, 5, 7], [0, 2, 4, 6, 8], [0, 0, 0, 0, 0], [0, -1, -3, -5, -7],
    [0, -2, -4, -6, -8]
  ]]];

  const poolModes: PoolMode[] = [undefined, 'max', 'avg'];
  const dataFormats: DataFormat[] =
      [undefined, 'channelsFirst', 'channelsLast'];
  const stridesArray = [1, 2];
  for (const poolMode of poolModes) {
    for (const dataFormat of dataFormats) {
      for (const stride of stridesArray) {
        const testTitle = `4x4, ${stride}, same, ${dataFormat}, ` +
            `${poolMode}`;
        it(testTitle, () => {
          let x: Tensor = tensor4d(x4by4Data, [1, 1, 4, 4]);
          if (dataFormat !== 'channelsFirst') {
            x = tfc.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
          }
          let yExpected: Tensor;
          if (poolMode === 'avg') {
            if (stride === 1) {
              yExpected = tensor4d(
                  [[[
                    [25, 45, 65, 75], [5, 5, 5, 5], [-25, -45, -65, -75],
                    [-30, -50, -70, -80]
                  ]]],
                  [1, 1, 4, 4]);
            } else {
              yExpected = tensor4d([[[[25, 65], [-25, -65]]]], [1, 1, 2, 2]);
            }
          } else {
            if (stride === 1) {
              yExpected = tensor4d(
                  [[[
                    [40, 60, 80, 80], [40, 60, 80, 80], [-10, -30, -50, -70],
                    [-20, -40, -60, -80]
                  ]]],
                  [1, 1, 4, 4]);
            } else if (stride === 2) {
              yExpected = tensor4d([[[[40, 80], [-10, -50]]]], [1, 1, 2, 2]);
            }
          }
          if (dataFormat !== 'channelsFirst') {
            yExpected = tfc.transpose(yExpected, [0, 2, 3, 1]);
          }
          const y =
              pool2d(x, [2, 2], [stride, stride], 'same', dataFormat, poolMode);
          expectTensorsClose(y, yExpected);
        });
      }
    }
  }

  for (const poolMode of poolModes) {
    it(`5x5, 2, same, CHANNEL_FIRST, ${poolMode}`, () => {
      const x5by5 = tensor4d(x5by5Data, [1, 1, 5, 5]);
      let yExpected = tensor4d(x4by4Data, [1, 1, 4, 4]);
      if (poolMode === 'avg') {
        yExpected = tensor4d(
            [[[[0.75, 4.5, 7.5], [-0.25, -2, -3.5], [-1, -5, -8]]]],
            [1, 1, 3, 3]);
      } else {
        yExpected =
            tensor4d([[[[2, 6, 8], [0, 0, 0], [0, -4, -8]]]], [1, 1, 3, 3]);
      }
      const y =
          pool2d(x5by5, [2, 2], [2, 2], 'same', 'channelsFirst', poolMode);
      expectTensorsClose(y, yExpected);
    });
  }

  for (const poolMode of poolModes) {
    it(`5x5, 2, valid, CHANNEL_LAST, ${poolMode}`, () => {
      const x5by5 =
          tfc.transpose(tensor4d(x5by5Data, [1, 1, 5, 5]), [0, 2, 3, 1]);
      let yExpected: Tensor4D;
      if (poolMode === 'avg') {
        yExpected = tensor4d([[[[0.75, 4.5], [-0.25, -2]]]], [1, 1, 2, 2]);
      } else {
        yExpected = tensor4d([[[[2, 6], [0, 0]]]], [1, 1, 2, 2]);
      }
      const y =
          pool2d(x5by5, [2, 2], [2, 2], 'valid', 'channelsLast', poolMode);
      expectTensorsClose(y, tfc.transpose(yExpected, [0, 2, 3, 1]));
    });
  }
});

describe('Pooling Layers 1D: Symbolic', () => {
  const poolSizes = [2, 3];
  const stridesList = [null, 1, 2];
  const poolModes: PoolMode[] = ['avg', 'max'];
  const paddingModes: PaddingMode[] = [undefined, 'valid', 'same'];

  for (const poolMode of poolModes) {
    for (const paddingMode of paddingModes) {
      for (const poolSize of poolSizes) {
        for (const strides of stridesList) {
          const testTitle = `poolSize=${poolSize}, ` +
              `${paddingMode}, ${poolMode}`;
          it(testTitle, () => {
            const inputLength = 16;
            const inputNumChannels = 11;
            const inputBatchSize = 2;
            const inputShape = [inputBatchSize, inputLength, inputNumChannels];
            const symbolicInput =
                new SymbolicTensor('float32', inputShape, null, [], null);
            const poolConstructor = poolMode === 'avg' ?
                tfl.layers.averagePooling1d :
                tfl.layers.maxPooling1d;
            const poolingLayer = poolConstructor({
              poolSize,
              strides,
              padding: paddingMode,
            });
            const output = poolingLayer.apply(symbolicInput) as SymbolicTensor;
            const expectedOutputLength = convOutputLength(
                inputLength, poolSize, paddingMode,
                strides ? strides : poolSize);
            const expectedShape =
                [inputBatchSize, expectedOutputLength, inputNumChannels];

            expect(output.shape).toEqual(expectedShape);
            expect(output.dtype).toEqual(symbolicInput.dtype);
          });
        }
      }
    }
  }
});

describeMathCPUAndGPU('Pooling Layers 1D: Tensor', () => {
  const poolModes = ['avg', 'max'];
  const strides = [2, 4];
  const poolSizes = [2, 4];
  const batchSize = 2;
  for (const poolMode of poolModes) {
    for (const stride of strides) {
      for (const poolSize of poolSizes) {
        const testTitle = `stride=${stride}, ${poolMode}, ` +
            `poolSize=${poolSize}`;
        it(testTitle, () => {
          const x2by8 = tensor2d([
            [10, 30, 50, 70, 20, 40, 60, 80],
            [-10, -30, -50, -70, -20, -40, -60, -80]
          ]);
          const x2by8by1 = tfc.expandDims(x2by8, 2);
          const poolConstructor = poolMode === 'avg' ?
              tfl.layers.averagePooling1d :
              tfl.layers.maxPooling1d;
          const poolingLayer = poolConstructor({
            poolSize,
            strides: stride,
            padding: 'valid',
          });
          const output = poolingLayer.apply(x2by8by1) as Tensor;
          let outputLength: number;
          let expectedOutputVals: number[][][];
          if (poolSize === 2) {
            if (stride === 2) {
              outputLength = 4;
              if (poolMode === 'avg') {
                expectedOutputVals =
                    [[[20], [60], [30], [70]], [[-20], [-60], [-30], [-70]]];
              } else {
                expectedOutputVals =
                    [[[30], [70], [40], [80]], [[-10], [-50], [-20], [-60]]];
              }
            } else if (stride === 4) {
              outputLength = 2;
              if (poolMode === 'avg') {
                expectedOutputVals = [[[20], [30]], [[-20], [-30]]];
              } else {
                expectedOutputVals = [[[30], [40]], [[-10], [-20]]];
              }
            }
          } else if (poolSize === 4) {
            if (stride === 2) {
              outputLength = 3;
              if (poolMode === 'avg') {
                expectedOutputVals =
                    [[[40], [45], [50]], [[-40], [-45], [-50]]];
              } else {
                expectedOutputVals =
                    [[[70], [70], [80]], [[-10], [-20], [-20]]];
              }
            } else if (stride === 4) {
              outputLength = 2;
              if (poolMode === 'avg') {
                expectedOutputVals = [[[40], [50]], [[-40], [-50]]];
              } else {
                expectedOutputVals = [[[70], [80]], [[-10], [-20]]];
              }
            }
          }
          const expectedShape: [number, number, number] =
              [batchSize, outputLength, 1];
          expectTensorsClose(
              output, tensor3d(expectedOutputVals, expectedShape));
        });
      }
    }
  }

  it('Handles poolSize and strides passed as number arrays', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.maxPool1d(
        {poolSize: [2], strides: [2], inputShape: [4, 3]}));
    const xs = tfc.ones([1, 4, 3]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(ys, [1, 1, 1, 1, 1, 1]);
    expect(ys.shape).toEqual([1, 2, 3]);

    const config = model.layers[0].getConfig();
    expect(config['poolSize']).toEqual([2]);
    expect(config['strides']).toEqual([2]);
  });
});

describe('Pooling Layers 2D: Symbolic', () => {
  const poolSizes = [2, 3];
  const poolModes = ['avg', 'max'];
  const paddingModes: PaddingMode[] = [undefined, 'valid', 'same'];
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];
  const poolSizeIsNumberValues = [false, true];

  for (const poolMode of poolModes) {
    for (const paddingMode of paddingModes) {
      for (const dataFormat of dataFormats) {
        for (const poolSize of poolSizes) {
          for (const poolSizeIsNumber of poolSizeIsNumberValues) {
            const testTitle = `poolSize=${poolSize}, ` +
                `${dataFormat}, ${paddingMode}, ` +
                `${poolMode}, ` +
                `poollSizeIsNumber=${poolSizeIsNumber}`;
            it(testTitle, () => {
              const inputShape = dataFormat === 'channelsFirst' ?
                  [2, 16, 11, 9] :
                  [2, 11, 9, 16];
              const symbolicInput =
                  new SymbolicTensor('float32', inputShape, null, [], null);

              const poolConstructor = poolMode === 'avg' ?
                  tfl.layers.averagePooling2d :
                  tfl.layers.maxPooling2d;
              const poolingLayer = poolConstructor({
                poolSize: poolSizeIsNumber ? poolSize : [poolSize, poolSize],
                padding: paddingMode,
                dataFormat,
              });

              const output =
                  poolingLayer.apply(symbolicInput) as SymbolicTensor;

              let outputRows = poolSize === 2 ? 5 : 3;
              if (paddingMode === 'same') {
                outputRows++;
              }
              let outputCols = poolSize === 2 ? 4 : 3;
              if (paddingMode === 'same' && poolSize === 2) {
                outputCols++;
              }

              let expectedShape: [number, number, number, number];
              if (dataFormat === 'channelsFirst') {
                expectedShape = [2, 16, outputRows, outputCols];
              } else {
                expectedShape = [2, outputRows, outputCols, 16];
              }

              expect(output.shape).toEqual(expectedShape);
              expect(output.dtype).toEqual(symbolicInput.dtype);
            });
          }
        }
      }
    }
  }

  const stridesValues: Array<number|[number, number]> = [1, [1, 1], [2, 1]];
  for (const strides of stridesValues) {
    it(`custom strides: ${strides}`, () => {
      const inputShape = [2, 16, 11, 3];
      const symbolicInput =
          new SymbolicTensor('float32', inputShape, null, [], null);
      const poolingLayer = tfl.layers.maxPooling2d({poolSize: [2, 2], strides});
      const output = poolingLayer.apply(symbolicInput) as tfl.SymbolicTensor;
      if (Array.isArray(strides) && util.arraysEqual(strides, [1, 1])) {
        expect(output.shape).toEqual([2, 15, 10, 3]);
      } else if (Array.isArray(strides) && util.arraysEqual(strides, [2, 1])) {
        expect(output.shape).toEqual([2, 8, 10, 3]);
      } else {
        // strides = 1
        expect(output.shape).toEqual([2, 15, 10, 3]);
      }
    });
  }

  it('Incorrect strides array length leads to error', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.layers.maxPooling2d({poolSize: 2, strides: [2]} as any))
        .toThrowError(/expected to have a length of 2/);
    expect(
        // tslint:disable-next-line:no-any
        () => tfl.layers.maxPooling2d({poolSize: 2, strides: [2, 3, 3]} as any))
        .toThrowError(/expected to have a length of 2/);
  });

  it('Invalid poolSize', () => {
    expect(() => tfl.layers.maxPooling2d({poolSize: 2.5, strides: 2}))
        .toThrowError(/poolSize.*positive integer.*2\.5\.$/);
    expect(() => tfl.layers.maxPooling2d({poolSize: 0, strides: 2}))
        .toThrowError(/poolSize.*positive integer.*0\.$/);
    expect(() => tfl.layers.maxPooling2d({poolSize: -2, strides: 2}))
        .toThrowError(/poolSize.*positive integer.*-2\.$/);
  });

  it('Invalid strides leads to Error', () => {
    expect(() => tfl.layers.maxPooling2d({poolSize: 3, strides: 2.5}))
        .toThrowError(/strides.*positive integer.*2\.5\.$/);
    expect(() => tfl.layers.maxPooling2d({poolSize: 3, strides: 0}))
        .toThrowError(/strides.*positive integer.*0\.$/);
    expect(() => tfl.layers.maxPooling2d({poolSize: 3, strides: -2}))
        .toThrowError(/strides.*positive integer.*-2\.$/);
  });
});

describeMathCPUAndGPU('Pooling Layers 2D: Tensor', () => {
  const x4by4Data =
      [10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80];

  const poolModes: PoolMode[] = ['avg', 'max'];
  const strides = [1, 2];
  const batchSizes = [2, 4];
  const channelsArray = [1, 3];
  for (const poolMode of poolModes) {
    for (const stride of strides) {
      for (const batchSize of batchSizes) {
        for (const channels of channelsArray) {
          const testTitle = `stride=${stride}, ${poolMode}, ` +
              `batchSize=${batchSize}, channels=${channels}`;
          it(testTitle, () => {
            let xArrayData: number[] = [];
            for (let b = 0; b < batchSize; ++b) {
              for (let c = 0; c < channels; ++c) {
                xArrayData = xArrayData.concat(x4by4Data);
              }
            }
            const x4by4 = tensor4d(xArrayData, [batchSize, channels, 4, 4]);

            const poolConstructor = poolMode === 'avg' ?
                tfl.layers.averagePooling2d :
                tfl.layers.maxPooling2d;
            const poolingLayer = poolConstructor({
              poolSize: [2, 2],
              strides: [stride, stride],
              padding: 'valid',
              dataFormat: 'channelsFirst',
            });

            const output = poolingLayer.apply(x4by4) as Tensor;

            let expectedShape: [number, number, number, number];
            let expectedOutputSlice: number[];
            if (poolMode === 'avg') {
              if (stride === 1) {
                expectedShape = [batchSize, channels, 3, 3];
                expectedOutputSlice = [25, 45, 65, 5, 5, 5, -25, -45, -65];
              } else if (stride === 2) {
                expectedShape = [batchSize, channels, 2, 2];
                expectedOutputSlice = [25, 65, -25, -65];
              }
            } else {
              if (stride === 1) {
                expectedShape = [batchSize, channels, 3, 3];
                expectedOutputSlice = [40, 60, 80, 40, 60, 80, -10, -30, -50];
              } else if (stride === 2) {
                expectedShape = [batchSize, channels, 2, 2];
                expectedOutputSlice = [40, 80, -10, -50];
              }
            }
            let expectedOutputArray: number[] = [];
            for (let b = 0; b < batchSize; ++b) {
              for (let c = 0; c < channels; ++c) {
                expectedOutputArray =
                    expectedOutputArray.concat(expectedOutputSlice);
              }
            }
            expectTensorsClose(
                output, tensor4d(expectedOutputArray, expectedShape));
          });
        }
      }
    }
  }

  it('Handles strides passed as number arrays', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.maxPooling2d(
        {poolSize: 2, strides: [2, 2], inputShape: [4, 4, 3]}));
    const xs = tfc.ones([1, 4, 4, 3]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(ys, tfc.ones([1, 2, 2, 3]));
    expect(ys.shape).toEqual([1, 2, 2, 3]);
    const config = model.layers[0].getConfig();
    expect(config['poolSize']).toEqual([2, 2]);
    expect(config['strides']).toEqual([2, 2]);
  });
});

describe('1D Global pooling Layers: Symbolic', () => {
  const globalPoolingLayers =
      [tfl.layers.globalAveragePooling1d, tfl.layers.globalMaxPooling1d];

  for (const globalPoolingLayer of globalPoolingLayers) {
    for (const hasArgs of [true, false]) {
      const testTitle = `layer=${globalPoolingLayer.name}; hasArgs=${hasArgs}`;
      it(testTitle, () => {
        const inputShape = [2, 11, 9];
        const symbolicInput =
            new SymbolicTensor('float32', inputShape, null, [], null);

        const layer = globalPoolingLayer(hasArgs ? {} : undefined);
        const output = layer.apply(symbolicInput) as SymbolicTensor;

        const expectedShape = [2, 9];
        expect(output.shape).toEqual(expectedShape);
        expect(output.dtype).toEqual(symbolicInput.dtype);
      });
    }
  }

  it('Invalid poolSize', () => {
    expect(() => tfl.layers.avgPooling1d({
      poolSize: 2.5
    })).toThrowError(/poolSize.*positive integer.*2\.5/);
    expect(() => tfl.layers.avgPooling1d({
      poolSize: 0
    })).toThrowError(/poolSize.*positive integer.*0\.$/);
    expect(() => tfl.layers.avgPooling1d({
      poolSize: -3
    })).toThrowError(/poolSize.*positive integer.*-3\.$/);
  });

  it('Invalid strides leads to Error', () => {
    expect(() => tfl.layers.avgPooling1d({poolSize: 3, strides: 4.5}))
        .toThrowError(/strides.*positive integer.*4\.5\.$/);
  });
});

describeMathCPUAndGPU('1D Global Pooling Layers: Tensor', () => {
  const x3DimData = [
    [[4, -1], [0, -2], [40, -10], [0, -20]],
    [[-4, 1], [0, 2], [-40, 10], [0, 20]]
  ];
  const globalPoolingLayers =
      [tfl.layers.globalAveragePooling1d, tfl.layers.globalMaxPooling1d];
  for (const globalPoolingLayer of globalPoolingLayers) {
    const testTitle = `globalPoolingLayer=${globalPoolingLayer.name}`;
    it(testTitle, () => {
      const x = tensor3d(x3DimData, [2, 4, 2]);
      const layer = globalPoolingLayer({});
      const output = layer.apply(x) as Tensor;

      let expectedOutput: Tensor2D;
      if (globalPoolingLayer === tfl.layers.globalAveragePooling1d) {
        expectedOutput = tensor2d([[11, -8.25], [-11, 8.25]], [2, 2]);
      } else {
        expectedOutput = tensor2d([[40, -1], [0, 20]], [2, 2]);
      }
      expectTensorsClose(output, expectedOutput);
    });
  }
});


describe('2D Global pooling Layers: Symbolic', () => {
  const globalPoolingLayers =
      [tfl.layers.globalAveragePooling2d, tfl.layers.globalMaxPooling2d];
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];

  for (const globalPoolingLayer of globalPoolingLayers) {
    for (const dataFormat of dataFormats) {
      const testTitle = `layer=${globalPoolingLayer.name}, ${dataFormat}`;
      it(testTitle, () => {
        const inputShape = [2, 16, 11, 9];
        const symbolicInput =
            new SymbolicTensor('float32', inputShape, null, [], null);

        const layer = globalPoolingLayer({dataFormat});
        const output = layer.apply(symbolicInput) as SymbolicTensor;

        const expectedShape = dataFormat === 'channelsLast' ? [2, 9] : [2, 16];
        expect(output.shape).toEqual(expectedShape);
        expect(output.dtype).toEqual(symbolicInput.dtype);
      });
    }
  }
});

describeMathCPUAndGPU('2D Global Pooling Layers: Tensor', () => {
  const x4DimData = [
    [[[4, -1], [0, -2]], [[40, -10], [0, -20]]],
    [[[4, -1], [0, -2]], [[40, -10], [0, -20]]]
  ];
  const dataFormats: DataFormat[] = ['channelsFirst', 'channelsLast'];

  const globalPoolingLayers =
      [tfl.layers.globalAveragePooling2d, tfl.layers.globalMaxPooling2d];
  for (const globalPoolingLayer of globalPoolingLayers) {
    for (const dataFormat of dataFormats) {
      const testTitle =
          `globalPoolingLayer=${globalPoolingLayer.name}, ${dataFormat}`;
      it(testTitle, () => {
        const x = tensor4d(x4DimData, [2, 2, 2, 2]);
        const layer = globalPoolingLayer({dataFormat});
        const output = layer.apply(x) as Tensor;

        let expectedOutput: Tensor2D;
        if (globalPoolingLayer === tfl.layers.globalAveragePooling2d) {
          if (dataFormat === 'channelsFirst') {
            expectedOutput = tensor2d([[0.25, 2.5], [0.25, 2.5]], [2, 2]);
          } else {
            expectedOutput = tensor2d([[11, -8.25], [11, -8.25]], [2, 2]);
          }
        } else {
          if (dataFormat === 'channelsFirst') {
            expectedOutput = tensor2d([[4, 40], [4, 40]], [2, 2]);
          } else {
            expectedOutput = tensor2d([[40, -1], [40, -1]], [2, 2]);
          }
        }
        expectTensorsClose(output, expectedOutput);
        const config = layer.getConfig();
        expect(config.dataFormat).toEqual(dataFormat);
      });
    }
  }
});
