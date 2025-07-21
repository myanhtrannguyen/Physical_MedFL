"""med: A Flower / PyTorch app for medical image segmentation."""

from collections import OrderedDict
from typing import Tuple, Dict
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from models.unet import UNet
from data_handling.data_loader import get_federated_dataloaders
from utils.metrics import evaluate_metrics

# Global variables
DATA_PATH = "../data/ACDC_preprocessed"  # Fixed path relative to med/ directory
N_CLASSES = 4
NUM_CLIENTS = 10
PARTITION_STRATEGY = 'non-iid'
TRAINING_SOURCES = ['slices']

# Cache for data loaders
_cached_dataloaders = None


def get_model():
    """Create and return UNet model."""
    return UNet(in_ch=1, num_classes=N_CLASSES)


def load_data(partition_id: int, num_partitions: int):
    """Load partition ACDC data."""
    global _cached_dataloaders
    
    # Only initialize dataloaders once
    if _cached_dataloaders is None:
        print(f"Loading federated dataloaders for {num_partitions} clients...")
        trainloaders, valloaders, testloader = get_federated_dataloaders(
            data_path=DATA_PATH,
            num_clients=num_partitions,
            batch_size=8,
            partition_strategy=PARTITION_STRATEGY,
            val_ratio=0.2,
            training_sources=TRAINING_SOURCES
        )
        _cached_dataloaders = (trainloaders, valloaders, testloader)
    
    trainloaders, valloaders, testloader = _cached_dataloaders
    
    if partition_id >= len(trainloaders):
        raise ValueError(f"Partition ID {partition_id} exceeds available partitions")
    
    return trainloaders[partition_id], valloaders[partition_id]


def get_testloader():
    """Get the global test loader."""
    global _cached_dataloaders
    
    if _cached_dataloaders is None:
        # Initialize if not done already
        load_data(0, NUM_CLIENTS)
    
    if _cached_dataloaders is not None:
        _, _, testloader = _cached_dataloaders
        return testloader
    else:
        return None


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)
    net.train()
    
    # Use appropriate loss for medical segmentation
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0, 1.0]).to(device))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    running_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
    
    avg_trainloss = running_loss / max(num_batches, 1)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    
    try:
        # Use comprehensive medical metrics
        metrics = evaluate_metrics(net, testloader, device, N_CLASSES)
        
        # Extract key metrics
        fg_dice_scores = metrics['dice_scores'][1:] if N_CLASSES > 1 else metrics['dice_scores']
        accuracy = float(np.mean(fg_dice_scores)) if len(fg_dice_scores) > 0 else 0.0
        loss = 1.0 - accuracy  # Convert dice to loss
        
        return loss, accuracy
    
    except Exception as e:
        print(f"Error in evaluation: {e}")
        # Fallback to simple evaluation
        criterion = nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        total_pixels = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss += criterion(outputs, labels.long()).item()
                
                # Calculate pixel accuracy
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total_pixels += labels.numel()
        
        accuracy = correct / max(total_pixels, 1)
        loss = loss / len(testloader)
        
        return loss, accuracy

def get_weights(net):
    """Extract model weights as numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Set model weights from numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
