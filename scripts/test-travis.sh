#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

# If this is nightly, use tfjs-core at master.
if [ "$TRAVIS_EVENT_TYPE" = cron ]
then
  git clone https://github.com/tensorflow/tfjs-core.git --depth=5
  cd tfjs-core
  yarn && yarn build && yarn publish-local
  cd ..
  yarn link-local '@tensorflow/tfjs-core'
fi

# Regular testing.
yarn build
yarn lint
yarn test-travis
