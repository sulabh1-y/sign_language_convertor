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

# Convert saved_model to tfjs_model directory containing model.json and binary shard files
echo "Converting saved_model to TensorFlow.js format..."
tensorflowjs_converter --input_format=tf_saved_model saved_model tfjs_model

echo "Conversion complete! Created directory 'tfjs_model/' with files:"
ls -la tfjs_model/
