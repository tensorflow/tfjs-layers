TypeScript Interfaces describing the Keras JSON format
------------------------------------------------------

This directory contains a description of the current Keras JSON serialization
format, in the form of TypeScript interfaces.  The intent is that any valid
Keras JSON file can be parsed in a type-safe manner using these types.

The Keras JSON format originated in the Python Keras implementation.  The basic
design is that the format mirrors the Python API.  Each class instance in a
Python model is serialized as a JSON object containing the class name and its
serialized constructor arguments.

The constructor arguments may be primitives, arrays of primitives, or dicts, in
which case the JSON serialization is straightforward.  If a constructor
argument is another object, then it is serialized in a nested manner.
Deserializing such a nested object configuration requires recursively
deserializing any object arguments, and finally calling the top-level
constructor.  In general this means that deserialization is purely tree-like, so
instances cannot be reused.  (The deserialization code for Models is an
exception to this principle, because it allows Layers to refer to each other in
order to describe a DAG).

There are several different kinds of configuration objects, currently
distinguished primarily by naming conventions.


FooBaseConfig
The subset of constructor arguments to Foo that are primitives (or arrays or dicts of primitives, etc.).  These arguments can be represented as JSON and also can be passed directly to the constructor.
FooConfig extends FooBaseConfig
In addition, provide fields for the constructor arguments that are not primitives, in nested JSON form (serialized).  The values of the non-primitive fields will always be *Serialization objects (below).
FooSerialization {
  class_name: 'Foo';
  config: FooConfig;
}
Bundle the config together with the class name that it is meant to serialize.
