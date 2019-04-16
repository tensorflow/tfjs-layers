#!/usr/bin/env bash

# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

cd integration_tests/tfjs2keras
./run-test.sh --stable
./run-test.sh --stable --tfkeras
./run-test.sh --dev --tfkeras
rm -r "test-data"
