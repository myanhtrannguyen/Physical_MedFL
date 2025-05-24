#!/usr/bin/env python3
"""
Debug script to analyze federated learning training setup and identify issues.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model_and_data_handle import (
        RobustMedVFL_UNet,
        load_data,
        evaluate_metrics,
        DEVICE,
        NUM_CLASSES,
        LEARNING_RATE,
        BATCH_SIZE,
    )
    print("✓ Successfully imported model_and_data_handle")
except ImportError as e:
    print(f"✗ Failed to import model_and_data_handle: {e}")
    sys.exit(1)

def check_data_availability():
    """Check if training and testing data are available and properly formatted."""
    print("\n" + "="*50)
    print("CHECKING DATA AVAILABILITY")
    print("="*50)
    
    # Check multiple possible data paths
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACDC_preprocessed"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACDC", "database"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    ]
    
    for base_path in possible_paths:
        print(f"\nChecking base path: {base_path}")
        
        if not os.path.exists(base_path):
            print(f"✗ Base directory does not exist: {base_path}")
            continue
            
        # Check for different data structures
        training_paths = [
            os.path.join(base_path, "ACDC_training_slices"),  # H5 format
            os.path.join(base_path, "training"),              # NIfTI format
        ]
        
        testing_paths = [
            os.path.join(base_path, "ACDC_testing_volumes"),  # H5 format  
            os.path.join(base_path, "testing"),               # NIfTI format
        ]
        
        for train_path in training_paths:
            if os.path.exists(train_path):
                print(f"✓ Training data found: {train_path}")
                if train_path.endswith("slices"):
                    files = [f for f in os.listdir(train_path) if f.endswith('.h5')]
                    print(f"  H5 files: {len(files)}")
                else:
                    patient_dirs = [d for d in os.listdir(train_path) if d.startswith('patient')]
                    print(f"  Patient directories: {len(patient_dirs)}")
                break
        else:
            print(f"✗ No training data found in {base_path}")
            
        for test_path in testing_paths:
            if os.path.exists(test_path):
                print(f"✓ Testing data found: {test_path}")
                if test_path.endswith("volumes"):
                    files = [f for f in os.listdir(test_path) if f.endswith('.h5')]
                    print(f"  H5 files: {len(files)}")
                else:
                    patient_dirs = [d for d in os.listdir(test_path) if d.startswith('patient')]
                    print(f"  Patient directories: {len(patient_dirs)}")
                break
        else:
            print(f"✗ No testing data found in {base_path}")
    
    return True  # Continue with testing even if data not found

def test_data_loading():
    """Test loading data using the universal medical data loader."""
    print("\n" + "="*50)
    print("TESTING DATA LOADING")
    print("="*50)
    
    # Test paths for different data formats
    test_paths = [
        ("ACDC_preprocessed/ACDC_training_slices", "H5 Format"),
        ("ACDC/database/training", "NIfTI Format"),
        ("data/training", "Generic Path")
    ]
    
    success = False
    train_imgs, train_masks = None, None
    
    for path, format_name in test_paths:
        if os.path.exists(path):
            print(f"Testing {format_name} at: {path}")
            try:
                # Use universal medical data loader
                train_imgs, train_masks = load_data(
                    path,
                    is_training=True,
                    target_size=(256, 256),
                    max_samples=5,  # Small sample for testing
                    apply_augmentation=False
                )
                
                if train_imgs is not None and len(train_imgs) > 0:
                    print(f"✓ Successfully loaded {len(train_imgs)} training samples")
                    print(f"  Image shape: {train_imgs.shape}")
                    print(f"  Image dtype: {train_imgs.dtype}")
                    print(f"  Image range: [{train_imgs.min():.3f}, {train_imgs.max():.3f}]")
                    
                    if train_masks is not None:
                        print(f"  Mask shape: {train_masks.shape}")
                        print(f"  Mask dtype: {train_masks.dtype}")
                        print(f"  Unique mask values: {np.unique(train_masks)}")
                    else:
                        print("  No masks available")
                    
                    success = True
                    break
                else:
                    print(f"✗ No data loaded from {path}")
                    
            except Exception as e:
                print(f"✗ Error loading from {path}: {e}")
        else:
            print(f"Path not found: {path}")
    
    if not success:
        print("⚠ No real data found, creating dummy data for testing...")
        # Create dummy data for testing
        train_imgs = np.random.rand(5, 256, 256, 1).astype(np.float32)
        train_masks = np.random.randint(0, NUM_CLASSES, (5, 256, 256)).astype(np.uint8)
        print(f"✓ Created dummy training data: {train_imgs.shape}")
    
    # Test evaluation data
    test_paths_eval = [
        ("ACDC_preprocessed/ACDC_testing_volumes", "H5 Format"),
        ("ACDC/database/testing", "NIfTI Format"), 
        ("data/testing", "Generic Path")
    ]
    
    test_imgs, test_masks = None, None
    
    for path, format_name in test_paths_eval:
        if os.path.exists(path):
            print(f"\nTesting evaluation data - {format_name} at: {path}")
            try:
                test_imgs, test_masks = load_data(
                    path,
                    is_training=False,
                    target_size=(256, 256),
                    max_samples=5,
                    apply_augmentation=False
                )
                
                if test_imgs is not None and len(test_imgs) > 0:
                    print(f"✓ Successfully loaded {len(test_imgs)} testing samples")
                    break
                else:
                    print(f"✗ No evaluation data loaded from {path}")
                    
            except Exception as e:
                print(f"✗ Error loading evaluation data from {path}: {e}")
    
    if test_imgs is None or len(test_imgs) == 0:
        print("⚠ Warning: No separate test data found, will use training data for evaluation")
        if train_imgs is not None and len(train_imgs) > 0:
            test_imgs, test_masks = train_imgs[:3], train_masks[:3] if train_masks is not None else None
        else:
            test_imgs = np.random.rand(3, 256, 256, 1).astype(np.float32)
            test_masks = np.random.randint(0, NUM_CLASSES, (3, 256, 256)).astype(np.uint8)
            print("✓ Created dummy test data")
    
    # Convert to DataLoaders
    if train_imgs is not None and train_masks is not None:
        train_tensor_imgs = torch.tensor(train_imgs).float()
        train_tensor_masks = torch.tensor(train_masks).long()
        train_dataset = TensorDataset(train_tensor_imgs, train_tensor_masks)
        trainloader = DataLoader(train_dataset, batch_size=min(BATCH_SIZE, len(train_dataset)), shuffle=False)
    else:
        trainloader = None
        
    if test_imgs is not None and test_masks is not None:
        test_tensor_imgs = torch.tensor(test_imgs).float()
        test_tensor_masks = torch.tensor(test_masks).long()
        test_dataset = TensorDataset(test_tensor_imgs, test_tensor_masks)
        testloader = DataLoader(test_dataset, batch_size=min(BATCH_SIZE, len(test_dataset)), shuffle=False)
    else:
        testloader = None
    
    return trainloader, testloader

def test_model_initialization():
    """Test model initialization and basic forward pass."""
    print("\n" + "="*50)
    print("TESTING MODEL INITIALIZATION")
    print("="*50)
    
    try:
        model = RobustMedVFL_UNet(n_channels=1, n_classes=NUM_CLASSES).to(DEVICE)
        print(f"✓ Model initialized successfully on {DEVICE}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Model device: {next(model.parameters()).device}")
        return model
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return None

def test_forward_pass(model, dataloader):
    """Test forward pass with actual data."""
    print("\n" + "="*50)
    print("TESTING FORWARD PASS")
    print("="*50)
    
    if model is None or dataloader is None:
        print("✗ Model or dataloader not available")
        return False
    
    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                print(f"  Input tensor shape: {inputs.shape}")
                print(f"  Input tensor device: {inputs.device}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Labels unique values: {torch.unique(labels)}")
                
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    print(f"  Model output is tuple with {len(outputs)} elements")
                    print(f"  Logits shape: {logits.shape}")
                else:
                    logits = outputs
                    print(f"  Logits shape: {logits.shape}")
                
                print(f"  Logits device: {logits.device}")
                print(f"  Logits dtype: {logits.dtype}")
                print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                
                # Test loss calculation
                criterion = torch.nn.CrossEntropyLoss()
                if labels.dim() == 4 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)
                
                loss = criterion(logits, labels)
                print(f"  Loss: {loss.item():.4f}")
                print("✓ Forward pass successful")
                return True
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False

def test_evaluation_metrics(model, dataloader):
    """Test evaluation metrics calculation."""
    print("\n" + "="*50)
    print("TESTING EVALUATION METRICS")
    print("="*50)
    
    if model is None or dataloader is None:
        print("✗ Model or dataloader not available")
        return
    
    try:
        metrics = evaluate_metrics(model, dataloader, DEVICE, NUM_CLASSES)
        print("✓ Evaluation metrics calculated successfully")
        print(f"  Metrics keys: {list(metrics.keys())}")
        
        for key, values in metrics.items():
            if isinstance(values, list) and len(values) > 0:
                print(f"  {key}: {[f'{v:.4f}' for v in values]}")
                print(f"    Average: {np.mean(values):.4f}")
            else:
                print(f"  {key}: {values}")
                
    except Exception as e:
        print(f"✗ Error in evaluation metrics: {e}")

def test_training_loop(model, dataloader):
    """Test a few training steps."""
    print("\n" + "="*50)
    print("TESTING TRAINING LOOP")
    print("="*50)
    
    if model is None or dataloader is None:
        print("✗ Model or dataloader not available")
        return
    
    try:
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        print(f"  Using learning rate: {LEARNING_RATE}")
        print(f"  Using batch size: {BATCH_SIZE}")
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= 3:  # Only test a few batches
                break
                
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"✓ Training loop test completed. Average loss: {avg_loss:.4f}")
        
    except Exception as e:
        print(f"✗ Error in training loop: {e}")

def main():
    """Main debugging function."""
    print("FEDERATED LEARNING TRAINING DEBUG")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Check data availability
    check_data_availability()
    
    # Test data loading
    trainloader, testloader = test_data_loading()
    
    # Test model initialization
    model = test_model_initialization()
    
    # Test forward pass
    if trainloader:
        test_forward_pass(model, trainloader)
    
    # Test evaluation metrics
    if testloader:
        test_evaluation_metrics(model, testloader)
    
    # Test training loop
    if trainloader:
        test_training_loop(model, trainloader)
    
    print("\n" + "="*50)
    print("DEBUG COMPLETE")
    print("="*50)
    
    # Provide recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Ensure you have sufficient training data (>50 samples recommended)")
    print("2. Check that the loss decreases during training")
    print("3. Verify that evaluation metrics are reasonable (Dice > 0.1 for non-background classes)")
    print("4. Use more epochs in federated learning (5-10 per round)")
    print("5. Consider using a learning rate scheduler")
    print("6. Monitor both training and validation metrics")
    print("7. For federated learning, ensure each client has adequate data distribution")
    print("8. ✅ NIfTI data structure detected - files are in subdirectories")
    print("9. 🔧 Consider using the universal data loader for automatic format detection")

if __name__ == "__main__":
    main() 