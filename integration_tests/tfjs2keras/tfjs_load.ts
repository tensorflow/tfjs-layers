/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';
import * as tfjsNode from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import {join} from 'path';

async function loadModel(modelDir: string) {
  const modelJsonPath = join(modelDir, 'model.json');
  const model = await tfl.loadModel(tfjsNode.io.fileSystem(modelJsonPath));

  // TODO(cais): Handle cases where there are multiple tensors in xs or ys.
  // Load xs values from JSON.
  const xs = tfc.tensor(
      JSON.parse(fs.readFileSync(join(modelDir, 'xs.json'), 'utf8')));
  // xs.print();  // DEBUG
  const ys = tfc.tensor(
      JSON.parse(fs.readFileSync(join(modelDir, 'ys.json'), 'utf8')));
  ys.print();  // DEBUG

  const modelOuts = model.predict(xs) as tfc.Tensor;
  modelOuts.print();  // DEBUG
  tfc.test_util.expectArraysClose(modelOuts, ys);
}

(async function main() {
  console.log(tfjsNode.version);

  if (process.argv.length !== 3) {
    console.error(` Expected 3 arguments,
      received $ {
    process.argv.length
  }
  `);
    process.exit(1);
  }
  const modelDir = process.argv[2];
  await loadModel(modelDir);
})();
