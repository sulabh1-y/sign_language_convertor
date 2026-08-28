#!/bin/bash
# convert_model.sh
# Script to convert trained Keras model (.h5) to TensorFlow.js format.

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Antigravity TF.js Model Converter ==="

# Check if tensorflowjs is installed
if ! pip3 show tensorflowjs &> /dev/null; then
    echo "tensorflowjs python package not found."
    echo "Installing tensorflowjs via pip..."
    pip3 install tensorflowjs
else
    echo "tensorflowjs package is already installed."
fi

# Force upgrade protobuf to resolve the incompatible runtime mismatch with yggdrasil-decision-forests
echo "Ensuring compatible protobuf runtime version..."
pip3 install "protobuf>=6.31.1"

# Convert model.h5 to tfjs_model directory containing model.json and binary shard files
echo "Converting model.h5 to TensorFlow.js format..."
KERAS_HOME=./.keras tensorflowjs_converter --input_format=keras model.h5 tfjs_model

# Patch Keras 3 InputLayer batch_shape mismatch and weights namespace for TF.js compatibility
echo "Patching model.json structure..."
python3 -c "
import json
d = json.load(open('tfjs_model/model.json'))
# Patch InputLayer
layers = d['modelTopology']['model_config']['config']['layers']
for l in layers:
    if l['class_name'] == 'InputLayer':
        l['config']['batchInputShape'] = l['config']['batch_shape']
# Patch weights manifest namespaces
weights = d['weightsManifest'][0]['weights']
for w in weights:
    name = w['name']
    if name.startswith('sequential/'):
        name = name[len('sequential/'):]
    if '/lstm_cell/' in name:
        name = name.replace('/lstm_cell/', '/')
    w['name'] = name
json.dump(d, open('tfjs_model/model.json', 'w'))
"

echo "Conversion complete! Created directory 'tfjs_model/' with files:"
ls -la tfjs_model/
