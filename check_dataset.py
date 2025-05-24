#!/usr/bin/env python3
"""
Check if the dataset can be loaded with the model    # Check model initialization with the data
    print("Checking model initialization...")
    try:
        # Assuming 1 channel for medical images and NUM_CLASSES output channels
        n_channels = images_train.shape[-1] if len(images_train.shape) >= 4 else 1
        model = RobustMedVFL_UNet(n_channels=n_channels, n_classes=NUM_CLASSES)
        print(f"Successfully initialized model with n_channels={n_channels}, n_classes={NUM_CLASSES}")
        
        # Create a simple tensor to test the forward pass
        dummy_input = torch.randn(2, n_channels, 256, 256)  # Adjust size as needed
        try:
            # The model's forward method returns a tuple with multiple outputs
            # (logits, physics_outputs_tuple, feature_maps)
            outputs = model(dummy_input)
            logits = outputs[0]  # First element is the segmentation logits
            print(f"Forward pass successful! Logits shape: {logits.shape}")
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            return False functions.
"""

import os
import sys
import torch
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from model_and_data_handle import (
        load_h5_data,
        RobustMedVFL_UNet,
        DEVICE,
        NUM_CLASSES,
        BATCH_SIZE
    )
    print("Successfully imported model_and_data_handle")
except ImportError as e:
    print(f"Error importing model_and_data_handle: {e}")
    sys.exit(1)

def check_dataset():
    """Check if the dataset exists and can be loaded."""
    base_dir = Path(current_dir) / "ACDC_preprocessed"
    
    if not base_dir.exists():
        print(f"Dataset directory not found: {base_dir}")
        print("Please run setup_dataset.py first.")
        return False
    
    # Check training data
    train_dir = base_dir / "ACDC_training_slices"
    if not train_dir.exists():
        print(f"Training data directory not found: {train_dir}")
        return False
        
    print(f"Attempting to load training data from {train_dir}")
    try:
        images_train, masks_train = load_h5_data(directory=str(train_dir), is_training=True, max_samples=2)
        if images_train is not None and masks_train is not None:
            print(f"Successfully loaded training data:")
            print(f"  - Images shape: {images_train.shape}")
            print(f"  - Masks shape: {masks_train.shape}")
        else:
            print("Failed to load training data (returned None)")
            return False
    except Exception as e:
        print(f"Error loading training data: {e}")
        return False
    
    # Check test data
    test_dir = base_dir / "ACDC_testing_volumes"
    if not test_dir.exists():
        print(f"Testing data directory not found: {test_dir}")
        return False
        
    print(f"Attempting to load testing data from {test_dir}")
    try:
        images_test, masks_test = load_h5_data(directory=str(test_dir), is_training=False, max_samples=1)
        if images_test is not None and masks_test is not None:
            print(f"Successfully loaded testing data:")
            print(f"  - Images shape: {images_test.shape}")
            print(f"  - Masks shape: {masks_test.shape}")
        else:
            print("Failed to load testing data (returned None)")
            return False
    except Exception as e:
        print(f"Error loading testing data: {e}")
        return False
    
    # Check model initialization with the data
    print("Checking model initialization...")
    try:
        # Assuming 1 channel for medical images and NUM_CLASSES output channels
        n_channels = images_train.shape[-1] if len(images_train.shape) >= 4 else 1
        model = RobustMedVFL_UNet(n_channels=n_channels, n_classes=NUM_CLASSES)
        print(f"Successfully initialized model with n_channels={n_channels}, n_classes={NUM_CLASSES}")
        
        # Create a simple tensor to test the forward pass
        dummy_input = torch.randn(2, n_channels, 256, 256)  # Adjust size as needed
        try:
            outputs = model(dummy_input)
            # Check if output is a tuple
            if isinstance(outputs, tuple):
                print(f"Forward pass successful! Output is a tuple with {len(outputs)} elements.")
                for i, out in enumerate(outputs):
                    if hasattr(out, 'shape'):
                        print(f"  - Output[{i}] shape: {out.shape}")
                    else:
                        print(f"  - Output[{i}] type: {type(out)}")
            else:
                print(f"Forward pass successful! Output shape: {outputs.shape}")
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            return False
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False
    
    print("\n✅ Dataset and model checks passed successfully!")
    print("You can now run the Flower simulation with the real dataset:")
    print("flower-simulation --server-app app_server:app --client-app app_client:app --num-supernodes 2 --app-dir .")
    return True

if __name__ == "__main__":
    check_dataset()
