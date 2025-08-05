#!/usr/bin/env python3
"""Debug script to test client_app loss function."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

print("🔍 Debugging client_app loss function...")

# Import and inspect client app
try:
    from med.client_app import MedicalFederatedClient
    print(f"✅ MedicalFederatedClient imported: {MedicalFederatedClient}")
    
    # Create a client instance to test
    client = MedicalFederatedClient(0)  # partition_id=0
    print(f"✅ Client created: {type(client)}")
    
    # Test _train_local_advanced method
    print("\n🔬 Inspecting _train_local_advanced method...")
    
    # Get method source
    import inspect
    try:
        source = inspect.getsource(client._train_local_advanced)
        print("📝 Source code snippet:")
        lines = source.split('\n')
        for i, line in enumerate(lines[:50]):  # First 50 lines
            if 'criterion' in line or 'CombinedLoss' in line or 'Adaptive' in line:
                print(f"Line {i+1}: {line.strip()}")
                
    except Exception as e:
        print(f"❌ Could not get source: {e}")
    
    print("\n🔧 Testing loss function instantiation...")
    
    # Test the loss initialization directly
    from utils.losses import CombinedLoss, Adaptive_tvmf_dice_loss
    import torch
    
    # Test what loss function is actually being used
    print("Testing CombinedLoss:")
    combined = CombinedLoss(num_classes=4, in_channels_maxwell=1024)
    print(f"✅ CombinedLoss: {type(combined)}")
    
    print("Testing Adaptive_tvmf_dice_loss:")
    adaptive = Adaptive_tvmf_dice_loss(num_classes=4)
    print(f"✅ Adaptive_tvmf_dice_loss: {type(adaptive)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
