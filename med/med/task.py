"""med: A Flower / PyTorch app for medical image segmentation."""

import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from models.unet import UNet
from data_handling.data_loader import get_federated_dataloaders
from utils.metrics import evaluate_metrics
from utils.losses import Adaptive_tvmf_dice_loss, DynamicWeightedLoss

# Global variables
N_CLASSES = 4
IMG_SIZE = 256
ALPHA = 0.5
NUM_WORKERS = 2
DATA_PATH = "../data/ACDC_preprocessed"
PARTITION_STRATEGY = "non-iid" 
TRAINING_SOURCES = ["slices"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model():
    """Load model architecture."""
    return UNet(in_ch=1, num_classes=N_CLASSES, base_c=32)

def get_weights(net) -> List[np.ndarray]:
    """Get model weights as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters: List[np.ndarray]) -> None:
    """Set model weights from a list of numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)

def load_data(partition_id: int, num_partitions: int):
    """Load partition data for federated learning."""
    logger.info(f"Loading federated dataloaders for {num_partitions} clients")
    
    trainloaders, valloaders, testloader = get_federated_dataloaders(
        data_path=DATA_PATH,
        num_clients=num_partitions,
        batch_size=8,
        partition_strategy=PARTITION_STRATEGY,
        val_ratio=0.2,
        alpha=ALPHA,
        training_sources=TRAINING_SOURCES,
        partition_by='patient',
        num_workers=NUM_WORKERS
    )
    
    logger.info("Dataloaders cached successfully with lazy loading")
    return trainloaders[partition_id], valloaders[partition_id], testloader

def train(net, trainloader, epochs, device, kappa_values=None):
    """Train the model on the training set with adaptive loss."""
    net.to(device)
    net.train()
    
    # Initialize loss components
    criterion = None
    base_criterion = None
    dynamic_weighter = None
    
    # Use advanced adaptive loss for medical segmentation
    if kappa_values:
        criterion = Adaptive_tvmf_dice_loss(
            num_classes=N_CLASSES,
            kappa_values=kappa_values
        ).to(device)
        logger.info(f"Using Adaptive t-vMF Dice Loss with kappa values: {kappa_values}")
    else:
        # Use weighted CrossEntropyLoss with DynamicWeightedLoss wrapper for adaptivity
        base_criterion = nn.CrossEntropyLoss(reduction='none')
        dynamic_weighter = DynamicWeightedLoss(
            num_classes=N_CLASSES,
            initial_weights=[0.1, 1.0, 1.0, 1.0]
        ).to(device)
        logger.info("Using CrossEntropyLoss with DynamicWeighted class adaptation")
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    running_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            
            if kappa_values and criterion is not None:
                # Advanced adaptive loss
                loss = criterion(outputs, labels.long())
            else:
                # Weighted CrossEntropy with dynamic adaptation
                assert base_criterion is not None and dynamic_weighter is not None
                base_losses = base_criterion(outputs, labels.long())
                # Convert to per-class loss for dynamic weighting
                class_losses = []
                for c in range(N_CLASSES):
                    class_mask = (labels == c)
                    if class_mask.sum() > 0:
                        class_losses.append(base_losses[class_mask].mean())
                    else:
                        class_losses.append(torch.tensor(0.0, device=device))
                loss = dynamic_weighter(class_losses)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
    
    avg_trainloss = running_loss / max(num_batches, 1)
    return avg_trainloss

def test(net, testloader, device):
    """Evaluate the model on the test set."""
    net.to(device)
    net.eval()
    
    try:
        # Use advanced metrics evaluation
        metrics = evaluate_metrics(net, testloader, device, N_CLASSES)
        
        # Calculate foreground metrics (excluding background class)
        fg_dice_scores = metrics['dice_scores'][1:] if N_CLASSES > 1 else metrics['dice_scores']
        avg_fg_dice = sum(fg_dice_scores) / len(fg_dice_scores) if fg_dice_scores else 0.0
        
        # Calculate loss using CrossEntropyLoss for compatibility
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels.long())
                total_loss += loss.item() * images.size(0)
                num_samples += images.size(0)
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        logger.info(f"Test evaluation - Loss: {avg_loss:.4f}, Avg FG Dice: {avg_fg_dice:.4f}")
        return avg_loss, avg_fg_dice
    
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        # Fallback to simple evaluation
        criterion = nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        total_pixels = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss += criterion(outputs, labels.long()).item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total_pixels += labels.numel()
        
        accuracy = correct / total_pixels if total_pixels > 0 else 0.0
        avg_loss = loss / len(testloader)
        
        logger.info(f"Fallback evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

def get_testloader():
    """Get test dataloader for server-side evaluation."""
    try:
        _, _, testloader = get_federated_dataloaders(
            data_path=DATA_PATH,
            num_clients=1,
            batch_size=8,
            partition_strategy="iid",
            val_ratio=0.2,
            training_sources=TRAINING_SOURCES,
            partition_by='patient',
            num_workers=NUM_WORKERS
        )
        return testloader
    except Exception as e:
        logger.error(f"Error loading test dataloader: {e}")
        return None
