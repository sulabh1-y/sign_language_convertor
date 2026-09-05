#!/bin/bash
# convert_model.sh
# Script to convert trained Keras model (.h5) to TensorFlow.js format.

set -e

echo "=== DeafBuddy TF.js Model Converter ==="

# Check if tensorflowjs is installed
if ! pip3 show tensorflowjs &> /dev/null; then
    echo "tensorflowjs python package not found."
    echo "Installing tensorflowjs via pip..."
    pip3 install tensorflowjs
else
    echo "tensorflowjs package is already installed."
fi

# Convert model.h5 to tfjs_model directory containing model.json and binary shard files
echo "Converting model.h5 to TensorFlow.js format..."
KERAS_HOME=./.keras tensorflowjs_converter --input_format=keras model.h5 tfjs_model

# Patch Keras 3 InputLayer batch_shape mismatch, DTypePolicy objects, and weight namespaces
echo "Sanitizing model.json structure for TensorFlow.js compatibility..."
python3 -c "
import json
d = json.load(open('tfjs_model/model.json'))

# 1. Patch InputLayer
layers = d['modelTopology']['model_config']['config']['layers']
for l in layers:
    if l['class_name'] == 'InputLayer':
        l['config']['batchInputShape'] = l['config']['batch_shape']

# 2. Patch Keras 3 DTypePolicy & Initializers recursively for TF.js
def sanitize_config(obj):
    if isinstance(obj, dict):
        if 'dtype' in obj and isinstance(obj['dtype'], dict):
            obj['dtype'] = obj['dtype'].get('config', {}).get('name', 'float32')
        for k, v in list(obj.items()):
            if k in ['kernel_initializer', 'recurrent_initializer', 'bias_initializer'] and isinstance(v, dict):
                v.pop('module', None)
                v.pop('registered_name', None)
                if 'config' in v and isinstance(v['config'], dict):
                    v['config'].pop('input_axes', None)
                    v['config'].pop('output_axes', None)
            sanitize_config(v)
    elif isinstance(obj, list):
        for item in obj:
            sanitize_config(item)

sanitize_config(d['modelTopology'])

# 3. Patch weights manifest namespaces
weights = d['weightsManifest'][0]['weights']
for w in weights:
    name = w['name']
    if name.startswith('sequential/'):
        name = name[len('sequential/'):]
    if '/lstm_cell/' in name:
        name = name.replace('/lstm_cell/', '/')
    w['name'] = name

json.dump(d, open('tfjs_model/model.json', 'w'), indent=2)
"

echo "Conversion complete! Created directory 'tfjs_model/' with files:"
ls -la tfjs_model/
