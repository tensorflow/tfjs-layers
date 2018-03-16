# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Run this as `yarn publish-npm`.

set -e

yarn build-npm
./scripts/make-version
npm publish
./scripts/tag-version
echo 'Yay! Published a new package to npm.'
