# tfjs-layers benchmarks

To run the benchmark script, first set up your environment.

(You may wish to set up python the requirements in a virtual environment.)

    pip install tensorflowjs


Once the development environment is prepared, execute the build script from the root of tfjs-layers.

    ./scripts/build-benchmarks-demo.sh

The script will benchmark the python construction and training of a number
of model architectures.  When it is complete, it will bring up a local HTTP
server.  Navigate to the local URL spcecified in stdout to bring up the
benchmarks page UI.  There will be a button to begin the JS side of the
benchmarks.  Clicking the button will run through and time the same models, now
running in the browser.

Once complete, the models' `fit()` and `predict()` costs are listed in a table.

Prese Ctl-c to end the http-server process.
