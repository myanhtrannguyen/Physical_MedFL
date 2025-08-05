#!/usr/bin/env python3
"""Simple test to check loss function in client_app."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

print("🔍 Testing client_app loss function...")

# Test imports
from utils.losses import CombinedLoss, Adaptive_tvmf_dice_loss
print("✅ Loss functions imported")

# Create instances
combined = CombinedLoss(num_classes=4, in_channels_maxwell=1024)
adaptive = Adaptive_tvmf_dice_loss(num_classes=4)

print(f"✅ CombinedLoss type: {type(combined)}")
print(f"✅ Adaptive type: {type(adaptive)}")

# Test physics parameters
import torch
x = torch.randn(2, 4, 128, 128)
y = torch.randint(0, 4, (2, 128, 128))
b1 = torch.randn(2, 4, 128, 128)
all_es = [torch.randn(2, 64, 128, 128)]
feat_sm = torch.randn(2, 4, 128, 128)

print("\n🧪 Testing loss functions...")

# Test CombinedLoss with physics
try:
    loss1 = combined(x, y.long(), b1=b1, all_es=all_es, feat_sm=feat_sm)
    print(f"✅ CombinedLoss with physics: {loss1.item():.4f}")
except Exception as e:
    print(f"❌ CombinedLoss error: {e}")

# Test Adaptive without physics
try:
    loss2 = adaptive(x, y.long())
    print(f"✅ Adaptive without physics: {loss2.item():.4f}")
except Exception as e:
    print(f"❌ Adaptive error: {e}")

# Test Adaptive with physics (should fail)
try:
    loss3 = adaptive(x, y.long(), b1=b1, all_es=all_es, feat_sm=feat_sm)
    print(f"❌ Adaptive should have failed with physics!")
except TypeError as e:
    print(f"✅ Adaptive correctly rejects physics: {str(e)[:50]}...")

print("\n📝 Checking client_app source...")
try:
    with open('med/client_app.py', 'r') as f:
        lines = f.readlines()
    
    criterion_lines = []
    for i, line in enumerate(lines):
        if 'criterion' in line and ('=' in line or 'CombinedLoss' in line or 'Adaptive' in line):
            criterion_lines.append(f"Line {i+1}: {line.strip()}")
    
    print("🔍 Criterion-related lines:")
    for line in criterion_lines[:10]:  # Show first 10
        print(f"  {line}")
        
except Exception as e:
    print(f"❌ Error reading file: {e}")

print("\n🎯 Summary: Client app should use CombinedLoss, not Adaptive_tvmf_dice_loss")
