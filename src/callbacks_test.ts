/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfl from './index';

fdescribe('EarlyStopping', () => {
  function createDummyModel(): tfl.LayersModel {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 1,
      inputShape: [1]
    }));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    return model;
  }

  it('Default monitor, default mode, increasing val_loss', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping();
    callback.setModel(model);

    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    await callback.onEpochEnd(2, {val_loss: 9.5});
    expect(model.stopTraining).toEqual(true);
  });

  it('Default monitor, default mode, holding val_loss', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping();
    callback.setModel(model);
  
    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    await callback.onEpochEnd(2, {val_loss: 9});
    expect(model.stopTraining).toEqual(true);
  });

  it('Custom monitor, default model, increasing', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping({monitor: 'aux_loss'});
    callback.setModel(model);
  
    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10, aux_loss: 100});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9, aux_loss: 120});
    expect(model.stopTraining).toEqual(true);
  });

  it('Custom monitor, max, increasing', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping({
      monitor: 'aux_metric',
      mode: 'max'
    });
    callback.setModel(model);
  
    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10, aux_metric: 100});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9, aux_metric: 120});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    await callback.onEpochEnd(2, {val_loss: 9, aux_metric: 110});
    expect(model.stopTraining).toEqual(true);
  });

  it('Patience = 2', async () => {
    const model = createDummyModel();
    const callback = tfl.callbacks.earlyStopping({patience: 2});
    callback.setModel(model);
  
    await callback.onTrainBegin();
    await callback.onEpochBegin(0);
    await callback.onEpochEnd(0, {val_loss: 10});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(1);
    await callback.onEpochEnd(1, {val_loss: 9});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(2);
    await callback.onEpochEnd(2, {val_loss: 9.5});
    expect(model.stopTraining).toBeUndefined();
    await callback.onEpochBegin(3);
    await callback.onEpochEnd(3, {val_loss: 9.6});
    expect(model.stopTraining).toEqual(true);
  });
});
