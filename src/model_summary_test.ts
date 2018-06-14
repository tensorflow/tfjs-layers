/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// Unit tests for tf.Model.summary() and tf.Sequential.summary().

import * as tfl from './index';
import {describeMathCPU} from './utils/test_utils';

function getRandomLayerOrModelName(length: number) {
  return 'L' + Math.random().toFixed(length - 1).slice(2);
}

describeMathCPU('Model.summary', () => {
  // let consoleLogHistory: string[];

  // beforeEach(() => {
  //   consoleLogHistory = [];
  //   spyOn(console, 'log').and.callFake((message?: string) => {
  //     consoleLogHistory.push(message);
  //   });
  // });

  // afterEach(() => {
  //   consoleLogHistory = [];
  // });

  it('Sequential model: one layer', () => {
    const layerName = getRandomLayerOrModelName(12);
    const model = tfl.sequential({
      layers:
          [tfl.layers.dense({units: 3, inputShape: [10], name: layerName})]
    });
    const lines = model.summary();
    expect(lines).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${layerName} (Dense)         [null,3]                  33        `,
      '=================================================================',
      'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
      '_________________________________________________________________'
    ]);
    // expect(consoleLogHistory).toEqual(lines);
  });

  it('Sequential model: one layer: custom lineLength', () => {
    const layerName = getRandomLayerOrModelName(12);
    const model = tfl.sequential({
      layers:
          [tfl.layers.dense({units: 3, inputShape: [10], name: layerName})]
    });
    const lines = model.summary(70);
    expect(lines).toEqual([
      '______________________________________________________________________',
      'Layer (type)                   Output shape                Param #    ',
      '======================================================================',
      `${layerName} (Dense)           [null,3]                    33         `,
      '======================================================================',
      'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
      '______________________________________________________________________'
    ]);
    // expect(consoleLogHistory).toEqual(lines);
  });

  it('Sequential model: one layer: custom positions', () => {
    const layerName = getRandomLayerOrModelName(12);
    const model = tfl.sequential({
      layers:
          [tfl.layers.dense({units: 3, inputShape: [10], name: layerName})]
    });
    const lines = model.summary(70, [0.5, 0.8, 1.0]);
    expect(lines).toEqual([
      '______________________________________________________________________',
      'Layer (type)                       Output shape         Param #       ',
      '======================================================================',
      `${layerName} (Dense)               [null,3]             33            `,
      '======================================================================',
      'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
      '______________________________________________________________________'
    ]);
  });

  it('Sequential model: one layer: custom printFn', () => {
    const layerName = getRandomLayerOrModelName(12);
    const model = tfl.sequential({
      layers:
          [tfl.layers.dense({units: 3, inputShape: [10], name: layerName})]
    });

    function muteLog(message?: any, ...optionalParams: any[]) {}

    const lines = model.summary(null, null, muteLog);
    expect(lines).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${layerName} (Dense)         [null,3]                  33        `,
      '=================================================================',
      'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
      '_________________________________________________________________'
    ]);
    // expect(consoleLogHistory).toEqual([]);
  });

  it('Sequential model: three layers', () => {
    const lyrName_1 = getRandomLayerOrModelName(12);
    const lyrName_2 = getRandomLayerOrModelName(12);
    const lyrName_3 = getRandomLayerOrModelName(12);
    const model = tfl.sequential({
      layers: [
        tfl.layers.flatten({inputShape: [2, 5], name: lyrName_1}),
        tfl.layers.dense({units: 3, name: lyrName_2}),
        tfl.layers.dense({units: 1, name: lyrName_3}),
      ]
    });
    const lines = model.summary();
    expect(lines).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${lyrName_1} (Flatten)       [null,10]                 0         `,
      '_________________________________________________________________',
      `${lyrName_2} (Dense)         [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName_3} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 37',
      'Non-trainable params: 0',
      '_________________________________________________________________',
    ]);
    // expect(consoleLogHistory).toEqual(lines);
  });

  it('Sequential model: with non-trainable layers', () => {
    const lyrName_1 = getRandomLayerOrModelName(12);
    const lyrName_2 = getRandomLayerOrModelName(12);
    const lyrName_3 = getRandomLayerOrModelName(12);
    const model = tfl.sequential({
      layers: [
        tfl.layers.flatten({inputShape: [2, 5], name: lyrName_1}),
        tfl.layers.dense({units: 3, name: lyrName_2, trainable: false}),
        tfl.layers.dense({units: 1, name: lyrName_3}),
      ]
    });
    const lines = model.summary();
    expect(lines).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${lyrName_1} (Flatten)       [null,10]                 0         `,
      '_________________________________________________________________',
      `${lyrName_2} (Dense)         [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName_3} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 4',
      'Non-trainable params: 33',
      '_________________________________________________________________',
    ]);
    // expect(consoleLogHistory).toEqual(lines);
  });

  it('Sequential model: nested', () => {
    const mdlName_1 = getRandomLayerOrModelName(12);
    const innerModel = tfl.sequential({
      layers: [tfl.layers.dense({units: 3, inputShape: [10]})],
      name: mdlName_1
    });
    const outerModel = tfl.sequential();
    outerModel.add(innerModel);

    const lryName_2 = getRandomLayerOrModelName(12);
    outerModel.add(tfl.layers.dense({units: 1, name: lryName_2}));

    const lines = outerModel.summary();
    expect(lines).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${mdlName_1} (Sequential)    [null,3]                  33        `,
      '_________________________________________________________________',
      `${lryName_2} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 37',
      'Non-trainable params: 0',
      '_________________________________________________________________',
    ]);
    // expect(consoleLogHistory).toEqual(lines);
  });
});
