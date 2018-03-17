# TensorFlow.js Layers: Easy-to-Use Machine Learning in the Browser

TensorFlow.js Layers is built on
[TensorFlow.js Core](https://github.com/tensorflow/tfjs-core), enabling users to
build, train and execute deep learning models in the browser with
an API that is easy-to-use and at a higher level of abstraction than
the core. TensorFlow.js Layers is largely compatible with
[Keras](https://keras.io/) and
[tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) and can
load saved models from those libraries.

## Installation

You can access TensorFlow.js through the
[@tensorflow/tfjs NPM package](https://www.npmjs.com/package/@tensorflow/tfjs-layers)

```sh
npm install @tensorflow/tfjs
```

To configure this in your `package.json`:

```js
"dependencies": {
  "@tensorflow/tfjs": "0.1.0"
}
```

## Getting started

### Building, training and executing a model

The following example shows how to build a toy model with only one `dense` layer
to perform linear regression.

```js
import * as tf from '@tensorflow/tfjs';

// A sequential model is a container which you can add layers to.
const model = tf.sequential();

// Add a dense layer with 1 output unit.
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Specify the loss type and optimizer for training.
model.compile({loss: 'meanSquaredError', optimizer: 'SGD'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train the model.
await model.fit(xs, ys, {epochs: 50});

// Ater the training, perform inference.
const output = await model.predict(tf.tensor2d([[5]], [1, 1]));
output.print();
```

### Loading a pretrained model

You can also load a model previously trained and saved from elsewhere (e.g.,
from Python Keras) and use it for inference or transfer learning in the browser.

For example, in Python, save your Keras model using `tensorflowjs.converter`
```python
import tensorflowjs as tfjs

# ... Create and train your Keras model.

# Save your Keras model in TensorFlow.js format.
tfjs.converter.save_keras_model(model, '/path/to/tfjs_artifacts/')

# Then use your favorite web server to serve the directory at a URL, say
#   http://foo.bar/tfjs_artifacts/model.json
```

To load the model with TensorFlow.js Layers:

```js
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadModel('http://foo.bar/tfjs_artifacts/model.json');
// Now the model is ready for inference, evaluation or re-training.
```

## For more information

- [TensorFlow.js API documentation](https://js.tensorflow.org/api/index.html)
- [TensorFlow.js Tutorials](https://js.tensorflow.org/tutorials/)
