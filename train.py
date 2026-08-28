#!/usr/bin/env python3
"""
train.py
Trains a sign language translator LSTM model.
If 'dataset.json' exists in the workspace, it loads and trains on custom recorded data.
Otherwise, it falls back to generating synthetic dummy data for demonstration.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def main():
    print("=== DeafBuddy LSTM Model Training ===")
    
    # 1. Configuration Constants
    sequence_length = 30  # Number of frames per gesture sequence
    num_features = 63     # 21 hand landmarks * 3 axes (X, Y, Z)
    dataset_file = 'dataset.json'
    
    # 2. Load Dataset (Custom or Synthetic)
    if os.path.exists(dataset_file):
        print(f"\n[REAL MODE] Custom dataset file '{dataset_file}' detected!")
        print("Loading coordinates...")
        
        with open(dataset_file, 'r') as f:
            raw_data = json.load(f)
        
        print(f"Successfully loaded {len(raw_data)} sequence samples.")
        
        # Parse samples
        x_list = []
        y_list = []
        skipped_samples = 0
        
        for idx, item in enumerate(raw_data):
            label = item.get('label')
            seq = item.get('sequence')
            
            # Validation checks
            if not label or not seq:
                print(f"  - Warning: Skipping corrupted sample at index {idx}.")
                skipped_samples += 1
                continue
                
            seq_np = np.array(seq, dtype=np.float32)
            
            # Verify shape match (30 frames, 63 coordinate values)
            if seq_np.shape != (sequence_length, num_features):
                print(f"  - Warning: Sample {idx} has invalid shape {seq_np.shape}. Expected ({sequence_length}, {num_features}). Skipping.")
                skipped_samples += 1
                continue
                
            x_list.append(seq_np)
            y_list.append(label)
            
        if skipped_samples > 0:
            print(f"Skipped {skipped_samples} malformed sample sequences.")
            
        if len(x_list) == 0:
            print("[ERROR] No valid data sequences found in dataset.json. Exiting.")
            return
            
        x_data = np.array(x_list, dtype=np.float32)
        
        # Encode Labels
        unique_classes = sorted(list(set(y_list)))
        num_classes = len(unique_classes)
        
        print(f"\nDetected {num_classes} unique gesture classes:")
        for idx, name in enumerate(unique_classes):
            count = y_list.count(name)
            print(f"  - Class [{idx}]: '{name}' ({count} samples)")
            
        # Write classes array copy-paste instructions for the frontend
        print("\n" + "=" * 65)
        print("💡 ATTENTION: COPY AND PASTE THIS INTO CONFIG.classes IN app.js:")
        print(f"classes: {unique_classes}")
        print("=" * 65 + "\n")
        
        # Map class names to integer labels
        y_labels = np.array([unique_classes.index(lbl) for lbl in y_list])
        y_data = to_categorical(y_labels, num_classes=num_classes)
        
    else:
        print(f"\n[DEMO MODE] Custom dataset file '{dataset_file}' NOT found.")
        print("💡 Tip: You can record real hand gestures using the custom builder UI")
        print("   on the frontend and download them to this folder as 'dataset.json'.")
        print("Generating synthetic dummy data for demonstration...")
        
        num_samples = 1000
        num_classes = 3
        unique_classes = ["Hello", "Thank You", "Goodbye"]
        
        x_data = np.random.rand(num_samples, sequence_length, num_features).astype(np.float32)
        y_labels = np.random.randint(0, num_classes, size=(num_samples,))
        y_data = to_categorical(y_labels, num_classes=num_classes)
        
        print(f"Generated {num_samples} random sequences of shape ({sequence_length}, {num_features}) for {num_classes} classes.")
    
    print("\nData Shape Summary:")
    print(f"  - X (Features matrix): {x_data.shape}")
    print(f"  - Y (Target matrix):   {y_data.shape}")
    
    # 3. Define LSTM Network Architecture
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # 4. Compile the Model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 5. Train the Model
    epochs = 15 if os.path.exists(dataset_file) else 5
    batch_size = 16 if os.path.exists(dataset_file) else 32
    val_split = 0.15 if len(x_data) >= 10 else 0.0 # prevent validation crash on micro datasets
    
    print(f"\nTraining model for {epochs} epochs (batch size: {batch_size})...")
    model.fit(
        x_data, 
        y_data, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=val_split,
        verbose=1
    )
    
    # 6. Export the Model as SavedModel
    model_dir = 'saved_model'
    print(f"\nExporting model to SavedModel directory '{model_dir}'...")
    model.export(model_dir)
    print("Saving model in H5 format...")
    model.save('model.h5')
    print("Model exported and saved successfully!")
    print("\nNext Step: Convert the model to TF.js format using:")
    print("  ./convert_model.sh")

if __name__ == '__main__':
    main()
