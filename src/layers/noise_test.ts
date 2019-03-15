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
 * Unit Tests for Noise Layers.
 */

import {describeMathCPU} from '../utils/test_utils';
import * as tfl from '../index';

describeMathCPU('GaussianNoise: Symbolic', () => {
  const stddevs = [0, 1, 0.5];
  const symbolicInputs = [
    new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
  ];

  for (const stddev of stddevs) {
    for (const symbolicInput of symbolicInputs) {
      const testTitle = `dropoutRate=${stddev}; ` +
        `input shape=${JSON.stringify(symbolicInput.shape)}`;
      it(testTitle, () => {
        const gaussianNoiseLayer = tfl.layers.gaussianNoise({stddev});
        const output =
          gaussianNoiseLayer.apply(symbolicInput) as tfl.SymbolicTensor;
        expect(output.dtype).toEqual(symbolicInput.dtype);
        expect(output.shape).toEqual(symbolicInput.shape);
        expect(output.sourceLayer).toEqual(gaussianNoiseLayer);
        expect(output.inputs).toEqual([symbolicInput]);
      });
    }
  }
});

describeMathCPU('GaussianDropout: Symbolic', () => {
  const rates = [0, 1, 0.5];
  const symbolicInputs = [
    new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
  ];

  for (const rate of rates) {
    for (const symbolicInput of symbolicInputs) {
      const testTitle = `dropoutRate=${rate}; ` +
        `input shape=${JSON.stringify(symbolicInput.shape)}`;
      it(testTitle, () => {
        const gaussianDropout = tfl.layers.gaussianDropout({rate});
        const output =
          gaussianDropout.apply(symbolicInput) as tfl.SymbolicTensor;
        expect(output.dtype).toEqual(symbolicInput.dtype);
        expect(output.shape).toEqual(symbolicInput.shape);
        expect(output.sourceLayer).toEqual(gaussianDropout);
        expect(output.inputs).toEqual([symbolicInput]);
      });
    }
  }
});

describeMathCPU('AlphaDropout: Symbolic', () => {
  const rates = [0, 1, 0.5];
  const symbolicInputs = [
    new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
    new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
  ];

  for (const rate of rates) {
    for (const symbolicInput of symbolicInputs) {
      const testTitle = `dropoutRate=${rate}; ` +
        `input shape=${JSON.stringify(symbolicInput.shape)}`;
      it(testTitle, () => {
        const alphaDropout = tfl.layers.alphaDropout({rate});
        const output =
          alphaDropout.apply(symbolicInput) as tfl.SymbolicTensor;
        expect(output.dtype).toEqual(symbolicInput.dtype);
        expect(output.shape).toEqual(symbolicInput.shape);
        expect(output.sourceLayer).toEqual(alphaDropout);
        expect(output.inputs).toEqual([symbolicInput]);
      });
    }
  }
});
