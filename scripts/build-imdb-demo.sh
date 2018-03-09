#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Builds the IMDB demo for TensorFlow.js Layers.
# Usage example: do under the root of the source repository:
#   ./scripts/build-imdb-demo.sh lstm
#
# Then open the demo HTML page in your browser, e.g.,
#   google-chrome demos/imdb_demo.html &

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# != 1 ]]; then
  echo "Usage:"
  echo "  build-imdb-demo.sh <MODEL_TYPE>"
  echo
  echo "MODEL_TYPE options: lstm | cnn"
  exit 1
fi
MODEL_TYPE=$1
echo "Using model type: ${MODEL_TYPE}"


# Build TensorFlow.js Layers standalone.
"${SCRIPTS_DIR}/build-standalone.sh"

DEMO_PATH="${SCRIPTS_DIR}/../dist/demo"
mkdir -p "${DEMO_PATH}"

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.
PYTHONPATH="${SCRIPTS_DIR}/.." python "${SCRIPTS_DIR}/imdb.py" \
    "${MODEL_TYPE}" \
    --metadata_json_path "${DEMO_PATH}/imdb.metadata.json" \
    --model_json_path "${DEMO_PATH}/imdb.keras.model.json" \
    --weights_json_path "${DEMO_PATH}/imdb.keras.weights.json"

# Prepend "const * = " to the json files.
printf "const imdbMetadataJSON = " > "${DEMO_PATH}/imdb.metadata.js"
cat "${DEMO_PATH}/imdb.metadata.json" >> "${DEMO_PATH}/imdb.metadata.js"
printf ";" >> "${DEMO_PATH}/imdb.metadata.js"
rm "${DEMO_PATH}/imdb.metadata.json"

printf "const imdbModelJSON = " > "${DEMO_PATH}/imdb.keras.model.js"
cat "${DEMO_PATH}/imdb.keras.model.json" >> "${DEMO_PATH}/imdb.keras.model.js"
printf ";" >> "${DEMO_PATH}/imdb.keras.model.js"
rm "${DEMO_PATH}/imdb.keras.model.json"

printf "const imdbWeightsJSON = " > "${DEMO_PATH}/imdb.keras.weights.js"
cat "${DEMO_PATH}/imdb.keras.weights.json" >> "${DEMO_PATH}/imdb.keras.weights.js"
printf ";" >> "${DEMO_PATH}/imdb.keras.weights.js"
rm "${DEMO_PATH}/imdb.keras.weights.json"

echo
echo "Now you can open the demo by:"
echo "  google-chrome demos/imdb_demo.html &"

