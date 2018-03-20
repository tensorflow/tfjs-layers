/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source: utils/generic_utils.py */

// tslint:disable:max-line-length
import {Tensor} from '@tensorflow/tfjs-core';
import * as _ from 'underscore';

import {AssertionError, AttributeError, IndexError, ValueError} from '../errors';
import {ConfigDict, ConfigDictValue, DType, Shape} from '../types';

// tslint:enable

/** Equivalent to Python's [value] * numValues */
// tslint:disable-next-line:no-any
export function pyListRepeat(value: any, numValues: number): any[] {
  if (Array.isArray(value)) {
    // tslint:disable-next-line:no-any
    let newArray: any[] = [];
    for (let i = 0; i < numValues; i++) {
      newArray = newArray.concat(value);
    }
    return newArray;
  } else {
    const newArray = new Array(numValues);
    newArray.fill(value);
    return newArray;
  }
}

/**
 * Equivalent to Python's getattr() built-in function.
 * @param obj
 * @param attrName The name of the attribute to retrieve.
 * @param defaultValue Default value to use if attrName doesn't exist in the
 *   object.
 */
// tslint:disable-next-line:no-any
export function pyGetAttr<T>(obj: any, attrName: string, defaultValue?: T): T {
  if (attrName in obj) {
    return obj[attrName];
  }
  if (_.isUndefined(defaultValue)) {
    throw new AttributeError(
        'pyGetAttr: Attempting to get attribute ' + attrName +
        'with no default value defined');
  }
  return defaultValue;
}

/**
 * Python allows indexing into a list from the end using negative values. This
 * utility functions translates an index into a list into a non-negative index,
 * allowing for negative indices, just like Python.
 *
 * @param x An array.
 * @param index The index to normalize.
 * @return A non-negative index, within range.
 * @exception IndexError if index is not within [-x.length, x.length)
 * @exception ValueError if x or index is null or undefined
 */
export function pyNormalizeArrayIndex<T>(x: T[], index: number): number {
  if (x == null || index == null) {
    throw new ValueError(
        `Must provide a valid array and index for ` +
        `pyNormalizeArrayIndex(). Got array ${x} and index ${index}.`);
  }
  const errMsg = `Index ${index} out of range for array of length ${x.length}`;
  if (index < 0) {
    if (index < -x.length) {
      throw new IndexError(errMsg);
    }
    return x.length + index;
  }
  if (index >= x.length) {
    throw new IndexError(errMsg);
  }
  return index;
}

export function assert(val: boolean, message?: string): void {
  if (!val) {
    throw new AssertionError(message);
  }
}

/**
 * Count the number of elements of the `array` that are equal to `reference`.
 */
export function count<T>(array: T[], refernce: T) {
  let counter = 0;
  for (const item of array) {
    if (item === refernce) {
      counter++;
    }
  }
  return counter;
}

/**
 * Type to represent class constructors.
 *
 * Source for this idea: https://stackoverflow.com/a/43607255
 */
// tslint:disable-next-line:no-any
export type Constructor<T> = new (...args: any[]) => T;

export class ClassNameMap {
  private static instance: ClassNameMap;
  // tslint:disable-next-line:no-any
  pythonClassNameMap: {[className: string]: any} = {};
  constructorClassNameMap: {[constructorName: string]: string} = {};

  static getMap() {
    if (ClassNameMap.instance == null) {
      ClassNameMap.instance = new ClassNameMap();
    }
    return ClassNameMap.instance;
  }

  static register<T>(className: string, cls: Constructor<T>) {
    this.getMap().pythonClassNameMap[className] =
        // tslint:disable-next-line:no-any
        [cls, (cls as any).fromConfig];
    this.getMap().constructorClassNameMap[cls.name] = className;
  }
}

export class SerializableEnumRegistry {
  private static instance: SerializableEnumRegistry;
  // tslint:disable-next-line:no-any
  enumRegistry: {[fieldName: string]: any};

  private constructor() {
    this.enumRegistry = {};
  }

  static getMap() {
    if (SerializableEnumRegistry.instance == null) {
      SerializableEnumRegistry.instance = new SerializableEnumRegistry();
    }
    return SerializableEnumRegistry.instance;
  }

  // tslint:disable-next-line:no-any
  static register(fieldName: string, enumCls: any) {
    if (SerializableEnumRegistry.contains(fieldName)) {
      throw new ValueError(
          `Attempting to register a repeated enum: ${fieldName}`);
    }
    this.getMap().enumRegistry[fieldName] = enumCls;
  }

  static contains(fieldName: string): boolean {
    return fieldName in this.getMap().enumRegistry;
  }

  // tslint:disable-next-line:no-any
  static lookup(fieldName: string, value: string): any {
    return this.getMap().enumRegistry[fieldName][value];
  }

  // tslint:disable-next-line:no-any
  static reverseLookup(fieldName: string, value: any): string {
    const enumMap = this.getMap().enumRegistry[fieldName];
    for (const candidateString in enumMap) {
      if (enumMap[candidateString] === value) {
        return candidateString;
      }
    }
    throw new ValueError(`Could not find serialization string for ${value}`);
  }
}

/**
 * If an array is of length 1, just return the first element. Otherwise, return
 * the full array.
 * @param tensors
 */
export function singletonOrArray<T>(xs: T[]): T|T[] {
  if (xs.length === 1) {
    return xs[0];
  }
  return xs;
}

/**
 * Normalizes a list/tensor into a list.
 *
 * If a tensor is passed, we return
 * a list of size 1 containing the tensor.
 *
 * @param x target object to be normalized.
 */
// tslint:disable-next-line:no-any
export function toList(x: any): any[] {
  if (Array.isArray(x)) {
    return x;
  }
  return [x];
}

/**
 * Generate a UID for a list
 */
// tslint:disable-next-line:no-any
export function objectListUid(objs: any|any[]): string {
  const objectList = toList(objs);
  let retVal = '';
  for (const obj of objectList) {
    if (obj.id == null) {
      throw new ValueError(
          `Object ${obj} passed to objectListUid without an id`);
    }
    if (retVal !== '') {
      retVal = retVal + ', ';
    }
    retVal = retVal + Math.abs(obj.id);
  }
  return retVal;
}
/**
 * Determine whether the input is an Array of Shapes.
 */
export function isArrayOfShapes(x: Shape|Shape[]): boolean {
  return Array.isArray(x) && Array.isArray(x[0]);
}

/**
 * Special case of normalizing shapes to lists.
 *
 * @param x A shape or list of shapes to normalize into a list of Shapes.
 * @return A list of Shapes.
 */
export function normalizeShapeList(x: Shape|Shape[]): Shape[] {
  if (x.length === 0) {
    return [];
  }
  if (!Array.isArray(x[0])) {
    return [x] as Shape[];
  }
  return x as Shape[];
}

/**
 * Checks whether an element or every element in a list is null or undefined.
 */
export function isAllNullOrUndefined(iterableOrElement: {}): boolean {
  return _.every(
      toList(iterableOrElement), x => (_.isNull(x) || _.isUndefined(x)));
}

/**
 * Converts string to snake-case.
 * @param name
 */
export function toSnakeCase(name: string): string {
  console.log(name, '......');
  const intermediate = name.replace(/(.)([A-Z][a-z0-9]+)/g, '$1_$2');
  const insecure =
      intermediate.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase();
  /*
   If the class is private the name starts with "_" which is not secure
   for creating scopes. We prefix the name with "private" in this case.
   */
  if (insecure[0] !== '_') {
    return insecure;
  }
  return 'private' + insecure;
}

export function toLowerCamelCase(identifier: string): string {
  // Split by upper case letters, numbers, or underscores.
  const words = identifier.split(/_|(?=[0-9A-Z])/).map((identifierWord, i) => {
    const firstChar = identifierWord.charAt(0);
    const restOfWord = identifierWord.length > 1 ?
        identifierWord.substring(1).toLowerCase() :
        '';

    return (i === 0 ? firstChar.toLowerCase() : firstChar.toUpperCase()) +
        restOfWord;
  });
  return words.join('');
}

// tslint:disable-next-line:no-any
let _GLOBAL_CUSTOM_OBJECTS = {} as {[objName: string]: any};

export function serializeKerasObject(
    // tslint:disable-next-line:no-any
    instance: any,
    constructorNameSymbolMap: {[constructorName: string]: string}):
    ConfigDictValue {
  if (instance === null || instance === undefined) {
    return null;
  }
  if (instance.getConfig != null) {
    if (constructorNameSymbolMap[instance.constructor.name] == null) {
      throw new ValueError(
          `Cannot find registry for ${instance.constructor.name}`);
    }
    return {
      className: constructorNameSymbolMap[instance.constructor.name],
      config: instance.getConfig()
    };
  }
  if (instance.name != null) {
    return instance.name;
  }
  throw new ValueError(`Cannot serialize ${instance}`);
}

/**
 * Deserialize a saved Keras Object
 * @param identifier either a string ID or a saved Keras dictionary
 * @param moduleObjects a list of Python class names to object constructors
 * @param customObjects a list of Python class names to object constructors
 * @param printableModuleName debug text for the object being reconstituted
 * @returns a TensorFlow.js Layers object
 */
// tslint:disable:no-any
export function deserializeKerasObject(
    identifier: string|ConfigDict,
    moduleObjects = {} as {[objName: string]: any},
    customObjects = {} as {[objName: string]: any},
    printableModuleName = 'object'): any {
  // tslint:enable
  if (typeof identifier === 'string') {
    const functionName = identifier;
    let fn;
    if (functionName in customObjects) {
      fn = customObjects[functionName];
    } else if (functionName in _GLOBAL_CUSTOM_OBJECTS) {
      fn = _GLOBAL_CUSTOM_OBJECTS[functionName];
    } else {
      fn = moduleObjects[functionName];
      if (fn == null) {
        throw new ValueError(`Unknown ${printableModuleName}: ${identifier}`);
      }
    }
    return fn;
  } else {
    // In this case we are dealing with a Keras config dictionary.
    const config = identifier;
    if (config.className == null || config.config == null) {
      throw new ValueError(
          `${printableModuleName}: Improper config format: ` +
          `${JSON.stringify(config)}.\n` +
          `'className' and 'config' must set.`);
    }
    const className = config.className as string;
    let cls, fromConfig;
    if (_.has(customObjects, className)) {
      [cls, fromConfig] = customObjects.get(className);
    } else if (_.has(_GLOBAL_CUSTOM_OBJECTS, className)) {
      [cls, fromConfig] = _GLOBAL_CUSTOM_OBJECTS.className;
    } else if (_.has(moduleObjects, className)) {
      [cls, fromConfig] = moduleObjects[className];
    }
    if (cls == null) {
      throw new ValueError(`Unknown ${printableModuleName}: ${className}`);
    }
    if (fromConfig != null) {
      // Porting notes: Instead of checking to see whether fromConfig accepts
      // customObjects, we create a customObjects dictionary and tack it on to
      // config.config as config.config.customObjects. Objects can use it, if
      // they want.

      // tslint:disable-next-line:no-any
      const customObjectsCombined = {} as {[objName: string]: any};
      for (const key of Object.keys(_GLOBAL_CUSTOM_OBJECTS)) {
        customObjectsCombined[key] = _GLOBAL_CUSTOM_OBJECTS[key];
      }
      for (const key of Object.keys(customObjects)) {
        customObjectsCombined[key] = customObjects[key];
      }
      // Add the customObjects to config
      const nestedConfig = config.config as ConfigDict;
      nestedConfig.customObjects = customObjectsCombined;

      const backupCustomObjects = {..._GLOBAL_CUSTOM_OBJECTS};
      for (const key of Object.keys(customObjects)) {
        _GLOBAL_CUSTOM_OBJECTS[key] = customObjects[key];
      }
      const returnObj = fromConfig(cls, config.config);
      _GLOBAL_CUSTOM_OBJECTS = {...backupCustomObjects};

      return returnObj;
    } else {
      // Then `cls` may be a function returning a class.
      // In this case by convention `config` holds
      // the kwargs of the function.
      const backupCustomObjects = {..._GLOBAL_CUSTOM_OBJECTS};
      for (const key of Object.keys(customObjects)) {
        _GLOBAL_CUSTOM_OBJECTS[key] = customObjects[key];
      }
      // In python this is **config['config'], for tfjs-layers we require
      // classes that use this fall-through construction method to take
      // a config interface that mimics the expansion of named parameters.
      const returnObj = new cls(config.config);
      _GLOBAL_CUSTOM_OBJECTS = {...backupCustomObjects};
      return returnObj;
    }
  }
}

/**
 * Helper function to obtain exactly one Tensor.
 * @param xs: A single `Tensor` or an `Array` of `Tensor`s.
 * @return A single `Tensor`. If `xs` is an `Array`, return the first one.
 * @throws ValueError: If `xs` is an `Array` and its length is not 1.
 */
export function getExactlyOneTensor(xs: Tensor|Tensor[]): Tensor {
  let x: Tensor;
  if (Array.isArray(xs)) {
    if (xs.length !== 1) {
      throw new ValueError(`Expected Tensor length to be 1; got ${xs.length}`);
    }
    x = xs[0];
  } else {
    x = xs as Tensor;
  }
  return x;
}

/**
 * Helper function to obtain exactly on instance of Shape.
 *
 * @param shapes Input single `Shape` or Array of `Shape`s.
 * @returns If input is a single `Shape`, return it unchanged. If the input is
 *   an `Array` containing exactly one instance of `Shape`, return the instance.
 *   Otherwise, throw a `ValueError`.
 * @throws ValueError: If input is an `Array` of `Shape`s, and its length is not
 *   1.
 */
export function getExactlyOneShape(shapes: Shape|Shape[]): Shape {
  if (Array.isArray(shapes) && Array.isArray(shapes[0])) {
    if (shapes.length === 1) {
      shapes = shapes as Shape[];
      return shapes[0];
    } else {
      throw new ValueError(`Expected exactly 1 Shape; got ${shapes.length}`);
    }
  } else {
    return shapes as Shape;
  }
}

/**
 * Compares two numbers for sorting.
 * @param a
 * @param b
 */
export function numberCompare(a: number, b: number) {
  return (a < b) ? -1 : ((a > b) ? 1 : 0);
}

/**
 * Comparison of two numbers for reverse sorting.
 * @param a
 * @param b
 */
export function reverseNumberCompare(a: number, b: number) {
  return -1 * numberCompare(a, b);
}

/**
 * Convert a string into the corresponding DType.
 * @param dtype
 * @returns An instance of DType.
 */
export function stringToDType(dtype: string): DType {
  switch (dtype) {
    case 'float32':
      return DType.float32;
    default:
      throw new ValueError(`Invalid dtype: ${dtype}`);
  }
}
