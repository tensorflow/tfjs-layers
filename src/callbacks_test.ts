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
 * Unit tests for callbacks.
 */

// tslint:disable:max-line-length
import {scalar} from '@tensorflow/tfjs-core';

import {disposeTensorsInLogs, resolveScalarsInLogs, UnresolvedLogs} from './engine/logs';
import {BaseLogger, History} from './engine/training';
import {describeMathCPUAndGPU} from './utils/test_utils';

// tslint:enable:max-line-length

describe('BaseLogger Callback', () => {
  it('Records and averages losses in an epoch', async done => {
    const baseLogger = new BaseLogger();
    baseLogger.setParams({metrics: ['loss', 'val_loss']});
    await baseLogger.onEpochBegin(0);
    await baseLogger.onBatchBegin(0);
    await baseLogger.onBatchEnd(0, {batch: 0, size: 10, loss: 5});
    await baseLogger.onBatchBegin(1);
    await baseLogger.onBatchEnd(1, {batch: 1, size: 10, loss: 6});
    await baseLogger.onBatchBegin(2);
    await baseLogger.onBatchEnd(2, {batch: 2, size: 5, loss: 7});
    const epochLog: UnresolvedLogs = {val_loss: 3};
    await baseLogger.onEpochEnd(0, epochLog);
    expect(epochLog['val_loss'] as number).toEqual(3);
    expect(epochLog['loss'] as number)
        .toBeCloseTo((10 * 5 + 10 * 6 + 5 * 7) / (10 + 10 + 5));
    done();
  });
  it('Forgets old epochs', async done => {
    const baseLogger = new BaseLogger();
    baseLogger.setParams({metrics: ['loss', 'val_loss']});
    const numOldEpochs = 2;
    for (let i = 0; i < numOldEpochs; ++i) {
      await baseLogger.onEpochBegin(i);
      await baseLogger.onBatchBegin(0);
      await baseLogger.onBatchEnd(0, {batch: 0, size: 10, loss: -5});
      const epochLog: UnresolvedLogs = {val_loss: 3};
      await baseLogger.onEpochEnd(i, epochLog);
    }
    await baseLogger.onEpochBegin(numOldEpochs);
    await baseLogger.onBatchBegin(0);
    await baseLogger.onBatchEnd(0, {batch: 0, size: 10, loss: 5});
    await baseLogger.onBatchBegin(1);
    await baseLogger.onBatchEnd(1, {batch: 1, size: 10, loss: 6});
    await baseLogger.onBatchBegin(2);
    await baseLogger.onBatchEnd(2, {batch: 2, size: 5, loss: 7});
    const epochLog: UnresolvedLogs = {val_loss: 3};
    await baseLogger.onEpochEnd(numOldEpochs, epochLog);
    expect(epochLog['val_loss'] as number).toEqual(3);
    expect(epochLog['loss'] as number)
        .toBeCloseTo((10 * 5 + 10 * 6 + 5 * 7) / (10 + 10 + 5));
    done();
  });
});

describe('History Callback', () => {
  it('onTrainBegin', async done => {
    const history = new History();
    await history.onTrainBegin();
    expect(history.epoch).toEqual([]);
    expect(history.history).toEqual({});
    done();
  });
  it('onEpochEnd', async done => {
    const history = new History();
    await history.onTrainBegin();
    await history.onEpochEnd(0, {'val_loss': 10, 'val_accuracy': 0.1});
    expect(history.epoch).toEqual([0]);
    expect(history.history).toEqual({'val_loss': [10], 'val_accuracy': [0.1]});
    await history.onEpochEnd(1, {'val_loss': 9.5, 'val_accuracy': 0.2});
    expect(history.epoch).toEqual([0, 1]);
    expect(history.history)
        .toEqual({'val_loss': [10, 9.5], 'val_accuracy': [0.1, 0.2]});
    done();
  });
});


describeMathCPUAndGPU('resolveScalarsInLogs', () => {
  it('Resolve mixed numbers and scalars', async done => {
    const logs: UnresolvedLogs = {
      'a': 1,
      'b': scalar(2),
      'c': -3,
      'd': scalar(-4),
    };
    await resolveScalarsInLogs(logs);
    expect(logs['a']).toEqual(1);
    expect(logs['b']).toEqual(2);
    expect(logs['c']).toEqual(-3);
    expect(logs['d']).toEqual(-4);
    done();
  });

  it('Resolve null works fine', async done => {
    const logs: UnresolvedLogs = null;
    await resolveScalarsInLogs(logs);
    expect(logs).toEqual(null);
    done();
  });

  it('Resolve empty works fine', async done => {
    const logs: UnresolvedLogs = {};
    await resolveScalarsInLogs(logs);
    expect(logs).toEqual({});
    done();
  });
});

describeMathCPUAndGPU('disposeTensorsInLogs', () => {
  it('Resolve mixed numbers and scalars', () => {
    const logs: UnresolvedLogs = {
      'a': 1,
      'b': scalar(2),
      'c': -3,
      'd': scalar(-4),
    };
    disposeTensorsInLogs(logs);
    expect(logs['a']).toEqual(1);
    // tslint:disable-next-line:no-any
    expect((logs['b'] as any).isDisposed).toEqual(true);
    expect(logs['c']).toEqual(-3);
    // tslint:disable-next-line:no-any
    expect((logs['d'] as any).isDisposed).toEqual(true);
  });
});
