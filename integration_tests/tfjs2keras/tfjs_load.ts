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

// The following call is done so that unhandled promises caused by
// exceptions inside async functions will lead to non-zero process
// exit codes and thereby cause the calling Python parent process
// to error out.
process.on('unhandledRejection', up => {throw up});

function readXsAndYsTensors(baseDir: string):
    {xs: tfc.Tensor[], ys: tfc.Tensor[]} {
  const XS_JSON_REGEX = /^xs_\d\.json$/;
  const YS_JSON_REGEX = /^ys_\d\.json$/;

  const outs = fs.readdirSync(baseDir);
  const xs: tfc.Tensor[] = [];
  const ys: tfc.Tensor[] = [];
  for (const out of outs) {
    if (out.match(XS_JSON_REGEX) || out.match(YS_JSON_REGEX)) {
      const shapePath =
          join(baseDir, out.slice(0, out.length - 5) + '.shape.json');
      const shape = JSON.parse(fs.readFileSync(shapePath, 'utf8'));
      const tensor =
          tfc.tensor(JSON.parse(fs.readFileSync(join(baseDir, out), 'utf8')))
              .reshape(shape);

      if (out.match(XS_JSON_REGEX)) {
        xs.push(tensor);
      } else {
        ys.push(tensor);
      }
    }
  }
  return {xs, ys};
}

async function loadModel(modelDir: string) {
  const modelJsonPath = join(modelDir, 'model.json');
  const model = await tfl.loadModel(tfjsNode.io.fileSystem(modelJsonPath));

  const {xs, ys} = readXsAndYsTensors(modelDir);

  let modelOuts = model.predict(xs) as tfc.Tensor | tfc.Tensor[];
  if (!Array.isArray(modelOuts)) {
    modelOuts = [modelOuts];
  }
  modelOuts = modelOuts as tfc.Tensor[];

  tfc.test_util.expectNumbersClose(modelOuts.length, ys.length);
  for (let i = 0; i < modelOuts.length; ++i) {
    tfc.test_util.expectArraysClose(modelOuts[i], ys[i]);
  }
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
