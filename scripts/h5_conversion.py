# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Library for converting from hdf5 to json + binary weights.

Used primarily to convert saved weights, or saved_models from their
hdf5 format to a JSON + binary weights format that the TS codebase can use.
."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile

import h5py
import keras
import numpy as np

import write_weights  # pylint: disable=import-error


class HDF5Converter(object):
  """Helper class to convert HDF5 format to JSON + binary weights format

  Used primaily to allow easy migration of a Python Keras trained
  and saved model to a JSON format for us in js based implementations.
  """
  def __init__(self, decimal_places=6):
    if decimal_places < 0:
      raise ValueError(
          'Expected decimal_places to be non-negative, but got %d' %
          decimal_places)
    self.decimal_places = decimal_places

  def _normalize_weight_name(self, weight_name):
    name = weight_name.decode('utf8')
    if name.endswith(':0'):
      # Python TensorFlow weight names ends with the output slot, which is
      # not applicable to TensorFlow.js.
      name = name[:-2]
    return name

  def convert_h5_group(self, group, names):
    """Construct a weights group entry.

    Args:
      group: The HDF5 group data.
      names: The names of the sub-fields within the group.

    Returns:
      An array of weight groups (see `write_weights` in TensorFlow.js).
    """
    if not names:
      return None
    names = [name.decode('utf8') for name in names]
    weight_values = [
        np.array(group[weight_name]) for weight_name in names]
    group_out = [{
        'name': self._normalize_weight_name(weight_name),
        'data': weight_value
    } for (weight_name, weight_value) in zip(names, weight_values)]
    return group_out

  def convert_h5_group_to_ascii(self, group, names):
    """Construct a JSON version of a HDF5 group.
    Args:
      group: The HDF5 group data.
      names: The names of the sub-fields within the group.
    """
    if not names:
      return None
    weight_values = [
        np.array(group[weight_name]) for weight_name in names]
    group_out = [{
        'name': self._normalize_weight_name(weight_name),
        'dtype': str(weight_value.dtype),
        'shape': list(weight_value.shape),
        'value': np.round(weight_value, self.decimal_places).tolist(),
    } for (weight_name, weight_value) in zip(names, weight_values)]
    return group_out

  def _check_version(self, h5file):
    """Check version compatiility.
    Args:
      h5file: An h5file object.
    Raises:
      ValueError: if the KerasVersion of the HDF5 file is unsupported.
    """
    keras_version = h5file.attrs['keras_version']
    if keras_version.split('.')[0] != '2':
      raise ValueError(
          'Expected Keras version 2; got Keras version %s' % keras_version)

  def _initialize_output_dictionary(self, h5file):
    """Prepopulate required fields for all data foramts.

    Args:
      h5file: Valid h5file object.

    Returns:
      A dictionary with common fields sets, shared across formats.
    """
    out = dict()
    out['keras_version'] = h5file.attrs['keras_version']
    out['backend'] = h5file.attrs['backend']
    return out

  def _ensure_h5file(self, h5file):
    if not isinstance(h5file, h5py.File):
      return h5py.File(h5file)
    else:
      return h5file

  def h5_merged_saved_model_to_tfjs_format(self, h5file):
    """Load topology & weight values from HDF5 file and convert.

    The HDF5 file is one generated by Keras' save_model method or model.save()

    N.B.:
    1) This function works only on HDF5 values from Keras version 2.
    2) This function does not perform conversion for special weights including
       ConvLSTM2D and CuDNNLSTM.

    Args:
      h5file: An instance of h5py.File, or the path to an h5py file.

    Returns:
      (model_json, groups)
        model_json: a JSON dictionary holding topology and system metadata.
        group: an array of group_weights as defined in tfjs write_weights.

    Raises:
      ValueError: If the Keras version of the HDF5 file is not supported.
    """
    h5file = self._ensure_h5file(h5file)
    self._check_version(h5file)
    model_json = self._initialize_output_dictionary(h5file)

    model_json['model_config'] = h5file.attrs['model_config']
    if 'training_config' in h5file.attrs:
      model_json['training_config'] = h5file.attrs['training_config']
    groups = []

    layer_names = [n.decode('utf8') for n in h5file['model_weights']]
    for layer_name in layer_names:
      layer = h5file['model_weights'][layer_name]
      group = self.convert_h5_group(
          layer, [name for name in layer.attrs['weight_names']])
      if group is not None:
        groups.append(group)
    return model_json, groups

  def h5_weights_to_tfjs_format(self, h5file):
    """Load weight values from a Keras HDF5 file and to a binary format.

    The HDF5 file is one generated by Keras' Model.save_weights() method.

    N.B.:
    1) This function works only on HDF5 values from Keras version 2.
    2) This function does not perform conversion for special weights including
       ConvLSTM2D and CuDNNLSTM.

    Args:
      h5file: An instance of h5py.File, or the path to an h5py file.

    Returns:
      An array of group_weights as defined in tfjs write_weights.

    Raises:
      ValueError: If the Keras version of the HDF5 file is not supported
    """
    h5file = self._ensure_h5file(h5file)
    self._check_version(h5file)
    groups = []

    # pylint: disable=not-an-iterable
    layer_names = [n.decode('utf8') for n in h5file.attrs['layer_names']]
    # pylint: enable=not-an-iterable
    for layer_name in layer_names:
      layer = h5file[layer_name]
      group = self.convert_h5_group(
          layer, [name for name in layer.attrs['weight_names']])
      if group is not None:
        groups.append(group)
    return groups

  def h5_merged_saved_model_to_json(self, h5file):
    """Load topology & weight values from HDF5 file and convert to JSON string.
    The HDF5 file is one generated by Keras' save_model method or model.save()
    N.B.:
    1) This function works only on HDF5 values from Keras version 2.
    2) This function does not perform conversion for special weights including
       ConvLSTM2D and CuDNNLSTM.
    Args:
      h5file: An instance of h5py.File, or the path to an h5py file.
    Raises:
      ValueError: If the Keras version of the HDF5 file is not supported, or if
        decimal_places is negative.
    """
    h5file = self._ensure_h5file(h5file)
    self._check_version(h5file)
    out = self._initialize_output_dictionary(h5file)
    out['model_config'] = h5file.attrs['model_config']
    if 'training_config' in h5file.attrs:
      out['training_config'] = h5file.attrs['training_config']
    out['model_weights'] = dict()
    out['optimizer_weights'] = dict()
    model_weights = out['model_weights']
    optimizer_weights = out['optimizer_weights']
    layer_names = [n.decode('utf8') for n in h5file['model_weights']]
    for layer_name in layer_names:
      layer = h5file['model_weights'][layer_name]
      model_weights[layer_name] = self.convert_h5_group_to_ascii(
          layer, [name for name in layer.attrs['weight_names']])
    if 'optimizer_weights' in h5file:
      optimizer_names = [n.decode('utf8') for n in h5file['optimizer_weights']]
      for optimizer_name in optimizer_names:
        optimizer = h5file['optimizer_weights'][optimizer_name]
        for key, value in optimizer.items():
          if isinstance(value, h5py.Dataset):
            optimizer_weights[optimizer_name] = [{
                'name' : self._normalize_weight_name(key),
                'dtype' : str(value.dtype),
                'shape' : list(value.shape),
                'value' :round(value.value, self.decimal_places)
            }]
          elif isinstance(value, h5py.Group):
            optimizer_weights[optimizer_name] = self.convert_h5_group_to_ascii(
                value, [name for name, _ in value.items()])
          else:
            print('Unknown h5py storage type in input file optimizer weights')
            print('key %s, value %s' % (key, value))
    return out

  def h5_weights_to_json(self, h5file):
    """Load weight values from a Keras HDF5 file and convert to a JSON string.
    The HDF5 file is one generated by Keras' Model.save_weights() method.
    N.B.:
    1) This function works only on HDF5 values from Keras version 2.
    2) This function does not perform conversion for special weights including
       ConvLSTM2D and CuDNNLSTM.
    Args:
      h5file: An instance of h5py.File, or the path to an h5py file.
    Returns:
      A JSON string encoding the loaded weights, in the format of the following
      example:
        {
          'keras_version': '2.1.2',
          'backend': 'tensorflow',
          'weights': {
            layer_name_1: [{
               name: foo,
               dtype: foo_dtype,
               shape: foo_shape,
               value: foo_value,
             },
             {
               name: bar,
               dtype: bar_dtype,
               shape: bar_shape,
               value: bar_values,
             }],
            layer_name_2: [{
                name: baz
                dtype: baz_dtype,
                shape: baz_shape,
                value: baz_value,
            }],
            ...
          },
        }
      The `dtype`s are represented as strings.
      The `shape`s are represented as Arrays of numbers.
      The `value`s are nested Arrays of numbers.
    Raises:
      ValueError: If the Keras version of the HDF5 file is not supported, or if
        decimal_places is negative.
    """
    h5file = self._ensure_h5file(h5file)
    self._check_version(h5file)
    out = self._initialize_output_dictionary(h5file)
    out['weights'] = dict()
    out_weights = out['weights']
    # pylint: disable=not-an-iterable
    layer_names = [n.decode('utf8') for n in h5file.attrs['layer_names']]
    # pylint: enable=not-an-iterable
    for layer_name in layer_names:
      layer = h5file[layer_name]
      out_weights[layer_name] = self.convert_h5_group_to_ascii(
          layer, [name for name in layer.attrs['weight_names']])
    return out

  def write_artifacts(self,
                      topology,
                      weights,
                      output_dir,
                      topology_filename='topology.json'):
    """Writes weights and topology to the output_dir.

    If `topology` is empty, only emit weights to output_dir.

    Args:
      topology: a JSON dictionary, representing the Keras config.
      weights: an array of weight groups (as defined in tfjs write_weights).
      output_dir: the directory to hold all the contents.
      topology_filename: filename for the topology json file.
    """
    # TODO(cais, nielsene): This method should allow optional arguments of
    #   `write_weights.write_weights` (e.g., shard size) and forward them.
    # We write the topology after since write_weights makes no promises about
    # preserving directory contents.
    if topology:
      json_path = os.path.join(output_dir, topology_filename)
      with open(json_path, 'wt') as json_file:
        json_file.write(json.dumps(topology))
    write_weights.write_weights(weights, output_dir)


def save_model(model, artifacts_dir):
  r"""Save a Keras model and its weigths in TensorFlow.js format.

  Args:
    model: An instance of `keras.Model`.
    artifacts_dir: The directory in which the artifacts will be saved.
      The artifacts to be saved include:
        - topology.json: A JSON representing the model. This JSON is obtained
          through calling `keras.models.save_model`.
        - weights_manifest.json: A TensorFlow.js-format JSON manifest for the
          model's weights.
        - files containing weight values in groups, with the file name pattern
          group(\d+)-shard(\d+)of(\d+).
      If the directory does not exist, this function will attempt to create it.

  Raises:
    ValueError: If `artifacts_dir` already exists as a file (not a directory).
  """
  temp_h5_path = tempfile.mktemp() + '.h5'
  keras.models.save_model(model, temp_h5_path)
  # TODO(cais): Maybe get rid of the class HDF5Converter to simplify the code.
  converter = HDF5Converter()
  topology_json, weights_group = (
      converter.h5_merged_saved_model_to_tfjs_format(temp_h5_path))
  if os.path.isfile(artifacts_dir):
    raise ValueError('Path "%s" already exists as a file.' % artifacts_dir)
  elif not os.path.isdir(artifacts_dir):
    os.makedirs(artifacts_dir)
  converter.write_artifacts(topology_json, weights_group, artifacts_dir)
  os.remove(temp_h5_path)
