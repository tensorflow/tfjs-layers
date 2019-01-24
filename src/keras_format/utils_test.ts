import {stringDictToArray} from './utils';

/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

class TestOptions {
  [key: string]: string;
  public readonly foo = 'Foo';
  public readonly bar = 'Bar';
  public readonly baz = 'Baz';
}

describe('stringDictToArray', () => {
  it('converts an object with string properties to an array of {value, label}.',
     () => {
       expect(stringDictToArray(new TestOptions())).toEqual([
         {value: 'foo', label: 'Foo'}, {value: 'bar', label: 'Bar'},
         {value: 'baz', label: 'Baz'}
       ]);
     });
});
