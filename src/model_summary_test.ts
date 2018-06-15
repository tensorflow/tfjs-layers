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
  let consoleLogHistory: string[];

  beforeEach(() => {
    consoleLogHistory = [];
    spyOn(console, 'log').and.callFake((message?: string) => {
      consoleLogHistory.push(message);
    });
  });

  afterEach(() => {
    consoleLogHistory = [];
  });

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
    expect(consoleLogHistory).toEqual(lines);
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
    expect(consoleLogHistory).toEqual(lines);
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
    expect(consoleLogHistory).toEqual(lines);
  });

  it('Sequential model: one layer: custom printFn', () => {
    const layerName = getRandomLayerOrModelName(12);
    const model = tfl.sequential({
      layers:
          [tfl.layers.dense({units: 3, inputShape: [10], name: layerName})]
    });

    const messages: string[] = [];
    // tslint:disable-next-line:no-any
    function rerouteLog(message?: any, ...optionalParams: any[]) {
      messages.push(message);
    }

    const lines = model.summary(null, null, rerouteLog);
    expect(lines).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${layerName} (Dense)         [null,3]                  33        `,
      '=================================================================',
      'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
      '_________________________________________________________________'
    ]);

    // console.log should have received no calls. But rerouteLog should have
    // received all the calls.
    expect(messages).toEqual(lines);
  });

  it('Sequential model: three layers', () => {
    const lyrName01 = getRandomLayerOrModelName(12);
    const lyrName02 = getRandomLayerOrModelName(12);
    const lyrName03 = getRandomLayerOrModelName(12);
    const model = tfl.sequential({
      layers: [
        tfl.layers.flatten({inputShape: [2, 5], name: lyrName01}),
        tfl.layers.dense({units: 3, name: lyrName02}),
        tfl.layers.dense({units: 1, name: lyrName03}),
      ]
    });
    const lines = model.summary();
    expect(lines).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${lyrName01} (Flatten)       [null,10]                 0         `,
      '_________________________________________________________________',
      `${lyrName02} (Dense)         [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName03} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 37',
      'Non-trainable params: 0',
      '_________________________________________________________________',
    ]);
    expect(consoleLogHistory).toEqual(lines);
  });

  it('Sequential model: with non-trainable layers', () => {
    const lyrName01 = getRandomLayerOrModelName(12);
    const lyrName02 = getRandomLayerOrModelName(12);
    const lyrName03 = getRandomLayerOrModelName(12);
    const model = tfl.sequential({
      layers: [
        tfl.layers.flatten({inputShape: [2, 5], name: lyrName01}),
        tfl.layers.dense({units: 3, name: lyrName02, trainable: false}),
        tfl.layers.dense({units: 1, name: lyrName03}),
      ]
    });
    const lines = model.summary();
    expect(lines).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${lyrName01} (Flatten)       [null,10]                 0         `,
      '_________________________________________________________________',
      `${lyrName02} (Dense)         [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName03} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 4',
      'Non-trainable params: 33',
      '_________________________________________________________________',
    ]);
    expect(consoleLogHistory).toEqual(lines);
  });

  it('Sequential model: nested', () => {
    const mdlName01 = getRandomLayerOrModelName(12);
    const innerModel = tfl.sequential({
      layers: [tfl.layers.dense({units: 3, inputShape: [10]})],
      name: mdlName01
    });
    const outerModel = tfl.sequential();
    outerModel.add(innerModel);

    const lyrName02 = getRandomLayerOrModelName(12);
    outerModel.add(tfl.layers.dense({units: 1, name: lyrName02}));

    const lines = outerModel.summary();
    expect(lines).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${mdlName01} (Sequential)    [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName02} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 37',
      'Non-trainable params: 0',
      '_________________________________________________________________',
    ]);
    expect(consoleLogHistory).toEqual(lines);
  });

  it('Functional model', () => {
    const lyrName01 = getRandomLayerOrModelName(12);
    const input1 = tfl.input({shape: [3], name: lyrName01});
    const lyrName02 = getRandomLayerOrModelName(12);
    const input2 = tfl.input({shape: [4], name: lyrName02});
    const lyrName03 = getRandomLayerOrModelName(12);
    const input3 = tfl.input({shape: [5], name: lyrName03});
    const lyrName04 = getRandomLayerOrModelName(12);
    const concat1 =
        tfl.layers.concatenate({name: lyrName04}).apply([input1, input2]) as
        tfl.SymbolicTensor;
    const lyrName05 = getRandomLayerOrModelName(12);
    const output =
        tfl.layers.concatenate({name: lyrName05}).apply([concat1, input3]) as
        tfl.SymbolicTensor;
    const model =
        tfl.model({inputs: [input1, input2, input3], outputs: output});

    const lines = model.summary(70);
    expect(lines).toEqual([
      '______________________________________________________________________',
      'Layer (type)           Output shape   Param # Recevies inputs         ',
      '======================================================================',
      `${lyrName01} (InputLay [null,3]       0                               `,
      '______________________________________________________________________',
      `${lyrName02} (InputLay [null,4]       0                               `,
      '______________________________________________________________________',
      `${lyrName04} (Concaten [null,7]       0       ${lyrName01}[0][0]      `,
      `                                              ${lyrName02}[0][0]      `,
      '______________________________________________________________________',
      `${lyrName03} (InputLay [null,5]       0                               `,
      '______________________________________________________________________',
      `${lyrName05} (Concaten [null,12]      0       ${lyrName04}[0][0]      `,
      `                                              ${lyrName03}[0][0]      `,
      '======================================================================',
      'Total params: 0', 'Trainable params: 0', 'Non-trainable params: 0',
      '______________________________________________________________________'
    ]);
    expect(consoleLogHistory).toEqual(lines);
  });
});
