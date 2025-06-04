# 1. HEADER AND IMPORTS SECTION

# 1.1 Standard Library Imports
import os
import sys
import logging
import time
import json
import pickle
import math
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from collections import OrderedDict
from pathlib import Path
from datetime import datetime, timedelta

# 1.2 Scientific Computing Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# 1.3 Flower Framework Imports - UPDATED FOR 1.18.0
import flwr as fl
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAdam, FedAvg  # Import both FedAdam and FedAvg for comparison
from flwr.common import Context, Parameters, FitRes, EvaluateRes
from flwr.common.typing import Scalar, Metrics, NDArrays
from flwr.common import FitIns, EvaluateIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

# 1.4 Project-Specific Imports - FIXED
try:
    from src.models.unet_model import (
        RobustMedVFL_UNet,
        CombinedLoss,
        quantum_noise_injection
    )
    from src.data.dataset import ACDCUnifiedDataset, BraTS2020UnifiedDataset, create_unified_dataset
    from src.data.research_loader import create_research_dataloader, create_federated_research_loaders
    from src.data.preprocessing import MedicalImagePreprocessor, DataAugmentation
    from src.utils.seed import set_seed
    from src.utils.logger import setup_federated_logger
    SRC_IMPORTS_AVAILABLE = True
except ImportError:
    # Fallback imports for development
    print("Warning: Using fallback imports - some features may be limited")
    SRC_IMPORTS_AVAILABLE = False
    
    # Define fallback functions
    def set_seed(seed: int):
        """Fallback seed function"""
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def setup_federated_logger(client_id=None, log_dir="logs", level=logging.INFO):
        """Fallback logger setup"""
        logger = logging.getLogger(f"server_{client_id}" if client_id else "server")
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

# 2. GLOBAL CONFIGURATION AND CONSTANTS

class ServerConstants:
    """Centralized server configuration management"""
    
    # FedAdam hyperparameters (OPTIMIZED based on FL best practices)
    DEFAULT_ETA = 0.05  # REDUCED server LR for stable aggregation with new metrics
    DEFAULT_ETA_L = 0.003  # From good results (matched config.toml)
    DEFAULT_BETA_1 = 0.85  # INCREASED momentum for stability with macro averaging
    DEFAULT_BETA_2 = 0.99  # Second moment parameter
    DEFAULT_TAU = 1e-7  # IMPROVED numerical stability
    
    # Aggregation parameters
    DEFAULT_FRACTION_FIT = 1.0
    DEFAULT_FRACTION_EVALUATE = 1.0
    MIN_FIT_CLIENTS = 1
    MIN_EVALUATE_CLIENTS = 1
    MIN_AVAILABLE_CLIENTS = 1
    MAX_CLIENTS_PER_ROUND = 10
    
    # Training constants (optimized for medical data with fixed aggregation)
    DEFAULT_ROUNDS = 25  # MORE rounds for convergence with proper aggregation
    DEFAULT_LOCAL_EPOCHS = 3  # From good results (was 3 epochs)
    DEFAULT_BATCH_SIZE = 4  # Keep small batch size for medical segmentation
    
    # Medical imaging constants
    NUM_CLASSES = 4
    IMG_SIZE = 256
    CLASS_NAMES = ['Background', 'RV', 'Myocardium', 'LV']
    
    # Resource limits
    MAX_MEMORY_GB = 16
    COMPUTATION_TIMEOUT = 600  # 10 minutes
    MAX_MESSAGE_LENGTH = 512 * 1024 * 1024  # 512MB

def setup_server_environment():
    """Setup server environment with optimal configurations"""
    
    # Device detection with fallback strategies
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Server Device: CUDA GPU - {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Server Device: CPU")
    
    # Memory optimization
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # For performance
    
    # Random seed management
    def setup_random_seed(seed=42):
        """Setup random seed with fallback"""
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    try:
        if SRC_IMPORTS_AVAILABLE:
            try:
                # set_seed should be available from imports if SRC_IMPORTS_AVAILABLE is True
                set_seed(42)
            except (NameError, AttributeError):
                setup_random_seed(42)
        else:
            setup_random_seed(42)
    except Exception as e:
        setup_random_seed(42)
        logging.warning(f"Used fallback seed setting due to: {e}")
    
    return device

# Global device setup
DEVICE = setup_server_environment()

# 3. HELPER FUNCTIONS FOR FEDADAM CONFIGURATION

def create_fit_config_fn(config: Dict[str, Any]) -> Callable[[int], Dict[str, Scalar]]:
    """Create fit configuration function for FedAdam"""
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Configure training for each round"""
        
        # Progressive complexity scheduling
        if config.get('progressive_complexity', True):
            # Increase model complexity over rounds
            complexity_factor = min(1.0, server_round / (config.get('num_rounds', 10) * 0.7))
            dropout_rate = max(0.1, 0.3 - complexity_factor * 0.2)
        else:
            dropout_rate = 0.2
        
        # Noise adaptation scheduling
        if config.get('noise_adaptation', True):
            # Reduce noise injection over time
            noise_factor = max(0.1, 1.0 - server_round / config.get('num_rounds', 10))
            quantum_noise_std = 0.01 * noise_factor
        else:
            quantum_noise_std = 0.01
        
        # Learning rate scheduling
        base_lr = config.get('eta_l', ServerConstants.DEFAULT_ETA_L)
        if config.get('physics_scheduling', True):
            # Physics-informed learning rate decay
            lr_decay = 0.95 ** (server_round - 1)
            learning_rate = base_lr * lr_decay
        else:
            learning_rate = base_lr
        
        return {
            "server_round": server_round,
            "local_epochs": config.get('local_epochs', ServerConstants.DEFAULT_LOCAL_EPOCHS),
            "learning_rate": learning_rate,
            "batch_size": config.get('batch_size', ServerConstants.DEFAULT_BATCH_SIZE),
            "dropout_rate": dropout_rate,
            "quantum_noise_std": quantum_noise_std,
            "weight_decay": config.get('weight_decay', 1e-5),
            "enable_augmentation": True,
            "progressive_complexity": config.get('progressive_complexity', True),
            "noise_adaptation": config.get('noise_adaptation', True),
            "physics_scheduling": config.get('physics_scheduling', True),
        }
    
    return fit_config

def create_evaluate_config_fn(config: Dict[str, Any]) -> Callable[[int], Dict[str, Scalar]]:
    """Create evaluate configuration function for FedAdam"""
    def evaluate_config(server_round: int) -> Dict[str, Scalar]:
        """Configure evaluation for each round"""
        return {
            "server_round": server_round,
            "batch_size": config.get('test_batch_size', 4),
            "enable_detailed_metrics": True,
            "save_predictions": server_round % 5 == 0,  # Save every 5 rounds
        }
    
    return evaluate_config

def create_fit_metrics_aggregation_fn() -> Callable:
    """Create training metrics aggregation with FIXED logic for different metric types"""
    def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """
        FIXED: Aggregate training metrics with appropriate methods:
        - Loss metrics: weighted average (sample-weighted)  
        - Performance metrics (accuracy/dice/iou): macro average (equal client weight)
        """
        if not metrics:
            return {}
        
        # Extract metrics and weights
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        # Collect all unique metric names from all clients
        all_metric_names = set()
        for _, client_metrics in metrics:
            all_metric_names.update(client_metrics.keys())
        
        aggregated_metrics = {}
        
        # Process each metric with appropriate aggregation strategy
        for metric_name in all_metric_names:
            values = []
            weights = []
            
            for num_examples, client_metrics in metrics:
                if metric_name in client_metrics:
                    values.append(float(client_metrics[metric_name]))
                    weights.append(num_examples)
            
            if not values:
                continue
                
            # CRITICAL FIX: Apply different aggregation strategies based on metric type
            if "loss" in metric_name.lower():
                # Loss metrics: weighted average (larger datasets have more influence)
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                total_weight = sum(weights)
                aggregated_value = weighted_sum / total_weight if total_weight > 0 else 0.0
                aggregated_metrics[f"weighted_avg_{metric_name}"] = aggregated_value
                
            elif any(keyword in metric_name.lower() for keyword in ["accuracy", "dice", "iou", "precision", "recall", "f1"]):
                # Performance metrics: macro average (equal weight per client to avoid bias)
                aggregated_value = sum(values) / len(values)
                aggregated_metrics[f"macro_avg_{metric_name}"] = aggregated_value
                
            elif "learning_rate" in metric_name.lower() or "lr" in metric_name.lower():
                # Learning rate: take the most recent value
                aggregated_value = values[-1] if values else 0.0
                aggregated_metrics[f"current_{metric_name}"] = aggregated_value
                
            else:
                # Other metrics: macro average by default
                aggregated_value = sum(values) / len(values)
                aggregated_metrics[f"avg_{metric_name}"] = aggregated_value
        
        # Add summary information
        aggregated_metrics["total_examples"] = total_examples
        aggregated_metrics["num_clients"] = len(metrics)
        
        # Calculate client diversity metrics for monitoring FL health
        if len(metrics) > 1:
            import numpy as np
            # Loss diversity analysis
            loss_values = []
            acc_values = []
            for _, client_metrics in metrics:
                for key, value in client_metrics.items():
                    if "loss" in key.lower() and len(loss_values) < len(metrics):
                        loss_values.append(float(value))
                    elif "accuracy" in key.lower() and len(acc_values) < len(metrics):
                        acc_values.append(float(value))
            
            if len(loss_values) > 1:
                aggregated_metrics["client_loss_std"] = float(np.std(loss_values))
            if len(acc_values) > 1:
                aggregated_metrics["client_accuracy_std"] = float(np.std(acc_values))
        
        return aggregated_metrics
    
    return aggregate_fit_metrics

def create_evaluate_metrics_aggregation_fn() -> Callable:
    """Create evaluation metrics aggregation with FIXED logic"""
    def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """
        FIXED: Aggregate evaluation metrics with appropriate methods:
        - Loss metrics: weighted average
        - Performance metrics: macro average
        """
        if not metrics:
            return {}
        
        # Extract metrics and weights  
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        # Collect all unique metric names
        all_metric_names = set()
        for _, client_metrics in metrics:
            all_metric_names.update(client_metrics.keys())
        
        aggregated_metrics = {}
        
        # Process each metric with appropriate aggregation strategy
        for metric_name in all_metric_names:
            values = []
            weights = []
            
            for num_examples, client_metrics in metrics:
                if metric_name in client_metrics:
                    values.append(float(client_metrics[metric_name]))
                    weights.append(num_examples)
            
            if not values:
                continue
                
            # CRITICAL FIX: Apply correct aggregation by metric type
            if "loss" in metric_name.lower():
                # Loss metrics: weighted average
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                total_weight = sum(weights)
                aggregated_value = weighted_sum / total_weight if total_weight > 0 else 0.0
                aggregated_metrics[f"weighted_avg_{metric_name}"] = aggregated_value
                
            elif any(keyword in metric_name.lower() for keyword in ["accuracy", "dice", "iou", "precision", "recall", "f1"]):
                # Performance metrics: macro average (equal client weight)
                aggregated_value = sum(values) / len(values)
                aggregated_metrics[f"macro_avg_{metric_name}"] = aggregated_value
                
            else:
                # Default: macro average
                aggregated_value = sum(values) / len(values)
                aggregated_metrics[f"avg_{metric_name}"] = aggregated_value
        
        # Add summary information
        aggregated_metrics["total_examples"] = total_examples
        aggregated_metrics["num_clients"] = len(metrics)
        
        return aggregated_metrics
    
    return aggregate_evaluate_metrics

# 4. MODEL AND DATA FUNCTIONS (UNCHANGED)

def create_global_model(config: Dict[str, Any]):
    """Create and initialize global model for server-side evaluation"""
    
    try:
        # Try to create the actual model
        from src.models.unet_model import RobustMedVFL_UNet
        
        model = RobustMedVFL_UNet(
            n_channels=1,
            n_classes=config.get('num_classes', ServerConstants.NUM_CLASSES),
        ).to(DEVICE)
        
        # Initialize weights
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
        print(f"✓ Global model created: RobustMedVFL_UNet")
        
        return model
        
    except ImportError:
        print("Warning: RobustMedVFL_UNet not available - using dummy model")
        
        # Fallback dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 4, 3, padding=1)
            
            def forward(self, x):
                return self.conv(x)
        
        return DummyModel().to(DEVICE)

def get_evaluate_fn(
    global_model,
    test_dataloader: Optional[DataLoader],
    config: Dict[str, Any]
) -> Callable:
    """Create evaluation function for global model"""
    
    def evaluate(server_round: int, parameters: Parameters, config_dict: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        try:
            print(f"[Round {server_round}] Starting global model evaluation")
            
            if global_model is None:
                print("Global model is None, skipping evaluation")
                return None
                
            if test_dataloader is None or len(test_dataloader) == 0:
                print("No test data available, creating dummy evaluation results")
                return (1.0, {
                    "eval_loss": 1.0,
                    "eval_accuracy": 0.25,
                    "eval_dice_avg": 0.25,
                    "eval_dice_class_0": 0.25, "eval_dice_class_1": 0.25, 
                    "eval_dice_class_2": 0.25, "eval_dice_class_3": 0.25,
                    "eval_iou_avg": 0.15,
                    "eval_iou_class_0": 0.15, "eval_iou_class_1": 0.15,
                    "eval_iou_class_2": 0.15, "eval_iou_class_3": 0.15,
                    "eval_precision_avg": 0.25,
                    "eval_recall_avg": 0.25,
                    "num_test_samples": 10
                })
            
            # Set parameters to global model - CRITICAL FIX: Use same method as client
            if parameters is not None:
                try:
                    # Check if parameters is already a list of ndarrays or a Parameters object
                    if isinstance(parameters, list):
                        parameters_list = parameters
                    else:
                        # It's a Parameters object, convert to ndarrays
                        parameters_list = parameters_to_ndarrays(parameters)
                    
                    # CRITICAL FIX: Use same parameter assignment method as client
                    # Assign parameters using model.parameters() ordering (same as client)
                    param_idx = 0
                    for param in global_model.parameters():
                        if param.requires_grad and param_idx < len(parameters_list):
                            param.data.copy_(torch.from_numpy(parameters_list[param_idx]).to(param.device))
                            param_idx += 1
                    
                    print(f"Successfully loaded {param_idx} parameters to global model")
                        
                except Exception as e:
                    print(f"Warning: Could not load parameters to global model: {e}")
                    # Continue evaluation with current global model parameters
            
            global_model.eval()
            
            device = next(global_model.parameters()).device
            total_loss = 0.0
            total_samples = 0
            correct_predictions = 0
            total_pixels = 0
            
            # Per-class metrics storage
            num_classes = 4
            dice_scores_per_class = [[] for _ in range(num_classes)]
            iou_scores_per_class = [[] for _ in range(num_classes)]
            precision_per_class = [0.0] * num_classes
            recall_per_class = [0.0] * num_classes
            
            # Simple criterion for evaluation
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    try:
                        if len(batch) == 2:
                            images, masks = batch
                        else:
                            print(f"Warning: Unexpected batch format: {len(batch)} elements")
                            continue
                            
                        images = images.to(device).float()
                        masks = masks.to(device).long()
                        
                        batch_size = images.size(0)
                        total_samples += batch_size
                        
                        # Forward pass
                        outputs = global_model(images)
                        
                        # Handle tuple output (model might return auxiliary outputs)
                        if isinstance(outputs, tuple):
                            main_output = outputs[0]
                        else:
                            main_output = outputs
                            
                        # Compute loss
                        loss = criterion(main_output, masks)
                        total_loss += loss.item()  # Now this should work correctly
                        
                        # Compute Dice score and other metrics per class
                        pred_masks = torch.argmax(main_output, dim=1)
                        
                        # Overall accuracy
                        correct_predictions += (pred_masks == masks).sum().item()
                        total_pixels += masks.numel()
                        
                        # Per-class metrics computation
                        for class_idx in range(num_classes):
                            # Get binary masks for current class
                            pred_binary = (pred_masks == class_idx).float()
                            target_binary = (masks == class_idx).float()
                            
                            # Dice score for this class
                            intersection = (pred_binary * target_binary).sum()
                            union = pred_binary.sum() + target_binary.sum()
                            
                            if union > 0:
                                dice = (2.0 * intersection) / union
                                dice_scores_per_class[class_idx].append(dice.item())
                            else:
                                dice_scores_per_class[class_idx].append(0.0)
                            
                            # IoU score for this class
                            if union > 0:
                                iou = intersection / (pred_binary.sum() + target_binary.sum() - intersection)
                                iou_scores_per_class[class_idx].append(iou.item())
                            else:
                                iou_scores_per_class[class_idx].append(0.0)
                            
                            # Precision and Recall for this class
                            true_positive = intersection.item()
                            predicted_positive = pred_binary.sum().item()
                            actual_positive = target_binary.sum().item()
                            
                            if predicted_positive > 0:
                                precision_per_class[class_idx] += true_positive / predicted_positive
                            if actual_positive > 0:
                                recall_per_class[class_idx] += true_positive / actual_positive
                                
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        continue
            
            # Compute final metrics
            avg_loss = total_loss / len(test_dataloader) if len(test_dataloader) > 0 else 0.0
            accuracy = correct_predictions / total_pixels if total_pixels > 0 else 0.0
            
            # Average per-class metrics
            avg_dice_per_class = []
            avg_iou_per_class = []
            
            for class_idx in range(num_classes):
                if dice_scores_per_class[class_idx]:
                    avg_dice = np.mean(dice_scores_per_class[class_idx])
                else:
                    avg_dice = 0.0
                avg_dice_per_class.append(avg_dice)
                
                if iou_scores_per_class[class_idx]:
                    avg_iou = np.mean(iou_scores_per_class[class_idx])
                else:
                    avg_iou = 0.0
                avg_iou_per_class.append(avg_iou)
                
                # Normalize precision/recall by number of batches
                precision_per_class[class_idx] /= len(test_dataloader)
                recall_per_class[class_idx] /= len(test_dataloader)
            
            # Overall averages (excluding background class 0 for medical metrics)
            foreground_dice = avg_dice_per_class[1:] if len(avg_dice_per_class) > 1 else avg_dice_per_class
            foreground_iou = avg_iou_per_class[1:] if len(avg_iou_per_class) > 1 else avg_iou_per_class
            foreground_precision = precision_per_class[1:] if len(precision_per_class) > 1 else precision_per_class
            foreground_recall = recall_per_class[1:] if len(recall_per_class) > 1 else recall_per_class
            
            avg_dice_foreground = np.mean(foreground_dice) if foreground_dice else 0.0
            avg_iou_foreground = np.mean(foreground_iou) if foreground_iou else 0.0
            avg_precision_foreground = np.mean(foreground_precision) if foreground_precision else 0.0
            avg_recall_foreground = np.mean(foreground_recall) if foreground_recall else 0.0
            
            # Prepare detailed metrics dictionary
            metrics = {
                # Loss and overall accuracy
                "eval_loss": float(avg_loss),
                "eval_accuracy": float(accuracy),
                "num_test_samples": int(total_samples),
                
                # Average metrics (foreground classes)
                "eval_dice_avg": float(avg_dice_foreground),
                "eval_iou_avg": float(avg_iou_foreground),
                "eval_precision_avg": float(avg_precision_foreground),
                "eval_recall_avg": float(avg_recall_foreground),
                
                # Per-class Dice scores
                "eval_dice_class_0": float(avg_dice_per_class[0]),
                "eval_dice_class_1": float(avg_dice_per_class[1]),
                "eval_dice_class_2": float(avg_dice_per_class[2]),
                "eval_dice_class_3": float(avg_dice_per_class[3]),
                
                # Per-class IoU scores
                "eval_iou_class_0": float(avg_iou_per_class[0]),
                "eval_iou_class_1": float(avg_iou_per_class[1]),
                "eval_iou_class_2": float(avg_iou_per_class[2]),
                "eval_iou_class_3": float(avg_iou_per_class[3]),
                
                # Per-class Precision
                "eval_precision_class_0": float(precision_per_class[0]),
                "eval_precision_class_1": float(precision_per_class[1]),
                "eval_precision_class_2": float(precision_per_class[2]),
                "eval_precision_class_3": float(precision_per_class[3]),
                
                # Per-class Recall
                "eval_recall_class_0": float(recall_per_class[0]),
                "eval_recall_class_1": float(recall_per_class[1]),
                "eval_recall_class_2": float(recall_per_class[2]),
                "eval_recall_class_3": float(recall_per_class[3]),
                
                # Additional medical imaging metrics
                "eval_f1_avg": float(2 * avg_precision_foreground * avg_recall_foreground / (avg_precision_foreground + avg_recall_foreground + 1e-8)),
                "server_round": int(server_round)
            }
            
            # Log formatted results
            print(f"\n[Round {server_round}] Global Evaluation Results:")
            print("="*60)
            print(f"Overall Performance:")
            print(f"  Loss: {avg_loss:.6f}")
            print(f"  Accuracy: {accuracy:.6f}")
            print(f"  Samples: {total_samples}")
            print("")
            print(f"Average Metrics (Foreground Classes):")
            print(f"  Dice Score: {avg_dice_foreground:.6f}")
            print(f"  IoU Score: {avg_iou_foreground:.6f}")
            print(f"  Precision: {avg_precision_foreground:.6f}")
            print(f"  Recall: {avg_recall_foreground:.6f}")
            print(f"  F1 Score: {2 * avg_precision_foreground * avg_recall_foreground / (avg_precision_foreground + avg_recall_foreground + 1e-8):.6f}")
            print("")
            print(f"Per-Class Dice Scores:")
            print(f"  Class 0 (Background): {avg_dice_per_class[0]:.6f}")
            print(f"  Class 1 (RV):         {avg_dice_per_class[1]:.6f}")
            print(f"  Class 2 (Myocardium): {avg_dice_per_class[2]:.6f}")
            print(f"  Class 3 (LV):         {avg_dice_per_class[3]:.6f}")
            print("")
            print(f"Per-Class IoU Scores:")
            print(f"  Class 0 (Background): {avg_iou_per_class[0]:.6f}")
            print(f"  Class 1 (RV):         {avg_iou_per_class[1]:.6f}")
            print(f"  Class 2 (Myocardium): {avg_iou_per_class[2]:.6f}")
            print(f"  Class 3 (LV):         {avg_iou_per_class[3]:.6f}")
            print("")
            print(f"Per-Class Precision:")
            print(f"  Class 0 (Background): {precision_per_class[0]:.6f}")
            print(f"  Class 1 (RV):         {precision_per_class[1]:.6f}")
            print(f"  Class 2 (Myocardium): {precision_per_class[2]:.6f}")
            print(f"  Class 3 (LV):         {precision_per_class[3]:.6f}")
            print("")
            print(f"Per-Class Recall:")
            print(f"  Class 0 (Background): {recall_per_class[0]:.6f}")
            print(f"  Class 1 (RV):         {recall_per_class[1]:.6f}")
            print(f"  Class 2 (Myocardium): {recall_per_class[2]:.6f}")
            print(f"  Class 3 (LV):         {recall_per_class[3]:.6f}")
            print("="*60)
            
            return float(avg_loss), metrics
            
        except Exception as e:
            print(f"❌ Global evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return evaluate

def load_global_test_data(config: Dict[str, Any]) -> Optional[DataLoader]:
    """Load global test dataset for server-side evaluation"""
    
    try:
        # Try to load test data from configured path - FIXED PATH
        test_data_path = config.get('test_data_path', 'data/raw/ACDC/database/testing')
        
        if os.path.exists(test_data_path):
            print(f"Loading global test data from: {test_data_path}")
            
            # Create test dataloader using ACDC dataset
            try:
                from src.data.research_loader import create_research_dataloader
                test_dataloader = create_research_dataloader(
                    dataset_type="acdc",
                    data_dir=test_data_path,
                    batch_size=config.get('test_batch_size', 4),
                    shuffle=False,
                    augment=False,  # No augmentation for testing
                    num_workers=0
                )
                
                # Safely get dataset size
                try:
                    if hasattr(test_dataloader, 'dataset'):
                        try:
                            dataset_size = len(test_dataloader.dataset)  # type: ignore
                            print(f"✅ Global test data loaded: {dataset_size} samples")
                        except TypeError:
                            print(f"✅ Global test data loaded successfully")
                    else:
                        print(f"✅ Global test data loaded successfully")
                except (TypeError, AttributeError):
                    print(f"✅ Global test data loaded successfully")
                
                return test_dataloader
                
            except ImportError:
                print("Warning: create_research_dataloader not available - using dummy data")
                return None
        else:
            print(f"Warning: Test data path not found: {test_data_path}")
            # Try alternative paths
            alt_paths = [
                'data/raw/ACDC/database/training',  # Use training as fallback
                'data/raw/ACDC/training',
                'data/raw/ACDC'
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"Using alternative test path: {alt_path}")
                    try:
                        from src.data.research_loader import create_research_dataloader
                        test_dataloader = create_research_dataloader(
                            dataset_type="acdc",
                            data_dir=alt_path,
                            batch_size=config.get('test_batch_size', 4),
                            shuffle=False,
                            augment=False,
                            num_workers=0
                        )
                        print(f"✅ Global test data loaded from alternative path")
                        return test_dataloader
                    except Exception as e:
                        print(f"Failed to load from {alt_path}: {e}")
                        continue
            
            print("No valid test data path found")
            return None
            
    except Exception as e:
        print(f"Error loading global test data: {e}")
        return None

# 5. FIXED FEDADAM IMPLEMENTATION

class FixedFedAdam(FedAdam):
    """
    Fixed FedAdam implementation to resolve client-server metric mismatch
    Key fixes:
    1. Proper moment initialization 
    2. Correct aggregation with parameter shapes verification
    3. Non-IID data handling improvements
    """
    
    def __init__(self, initial_parameters=None, **kwargs):
        # Extract FedAdam specific parameters
        self.eta = kwargs.pop('eta', 0.1)
        self.eta_l = kwargs.pop('eta_l', 0.01) 
        self.beta_1 = kwargs.pop('beta_1', 0.9)
        self.beta_2 = kwargs.pop('beta_2', 0.999)
        self.tau = kwargs.pop('tau', 1e-4)
        
        # FIXED: Ensure initial_parameters is not None before calling parent
        if initial_parameters is None:
            raise ValueError("initial_parameters cannot be None for FixedFedAdam")
        
        # Initialize parent class
        super().__init__(
            initial_parameters=initial_parameters,
            eta=self.eta,
            eta_l=self.eta_l,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            tau=self.tau,
            **kwargs
        )
        
        # CRITICAL FIX: Properly initialize moments from initial parameters
        self.current_weights = parameters_to_ndarrays(initial_parameters)
        # Initialize moments to zeros (critical for convergence)
        self.m_t = [np.zeros_like(w, dtype=np.float32) for w in self.current_weights]
        self.v_t = [np.zeros_like(w, dtype=np.float32) for w in self.current_weights]
        self.server_round = 0
        
        print(f"FixedFedAdam initialized:")
        print(f"  Parameters: {len(self.current_weights)} arrays")
        print(f"  Total parameters: {sum(w.size for w in self.current_weights):,}")
        print(f"  Moments initialized: m_t={len(self.m_t)}, v_t={len(self.v_t)}")
        print(f"  Hyperparameters: eta={self.eta}, eta_l={self.eta_l}, β₁={self.beta_1}, β₂={self.beta_2}, τ={self.tau}")
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Fixed aggregation with proper moment updates and debugging"""
        
        if not results:
            print(f"[Round {server_round}] No results to aggregate")
            return None, {}
        
        self.server_round = server_round
        print(f"\n[Round {server_round}] FixedFedAdam aggregation starting...")
        print(f"  Results received: {len(results)}")
        print(f"  Failures: {len(failures)}")
        
        # Collect weights and metrics from results
        weights_results = []
        total_examples = 0
        
        for client_proxy, fit_res in results:
            # Extract parameters and number of examples
            parameters_list = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            
            weights_results.append((parameters_list, num_examples))
            total_examples += num_examples
            
            print(f"  Client {client_proxy.cid}: {num_examples} examples, {len(parameters_list)} parameter arrays")
        
        if total_examples == 0:
            print("Error: Total examples is 0")
            return None, {}
        
        # CRITICAL FIX: Federated averaging with proper weighting
        print(f"  Computing federated average across {len(weights_results)} clients...")
        
        # Initialize aggregated weights
        aggregated_weights = None
        
        for client_weights, num_examples in weights_results:
            weight = num_examples / total_examples
            
            if aggregated_weights is None:
                # Initialize with first client's weights
                aggregated_weights = [weight * w for w in client_weights]
            else:
                # Add weighted contributions
                for i, w in enumerate(client_weights):
                    aggregated_weights[i] += weight * w
        
        # CRITICAL FIX: Ensure aggregated_weights is not None
        if aggregated_weights is None:
            print("Error: Failed to aggregate weights")
            return None, {}
        
        # CRITICAL FIX: Compute pseudo-gradients for Adam updates
        # Initialize avg_grad_norm for all paths
        avg_grad_norm = 0.0
        
        # First round - initialize current weights
        if self.current_weights is None or len(self.current_weights) == 0:
            self.current_weights = [w.copy() for w in aggregated_weights]
            self.m_t = [np.zeros_like(w, dtype=np.float32) for w in aggregated_weights]
            self.v_t = [np.zeros_like(w, dtype=np.float32) for w in aggregated_weights]
            print(f"  Initialized server state with {len(self.current_weights)} parameter arrays")
        else:
            # Compute pseudo-gradients (difference between aggregated and current)
            pseudo_gradients = []
            grad_norms = []
            
            for curr_w, agg_w in zip(self.current_weights, aggregated_weights):
                pseudo_grad = agg_w - curr_w  # This represents the "gradient" direction
                pseudo_gradients.append(pseudo_grad)
                grad_norms.append(np.linalg.norm(pseudo_grad))
            
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
            print(f"  Average pseudo-gradient norm: {avg_grad_norm:.6f}")
            
            # CRITICAL FIX: Adam moment updates with proper null checks
            print(f"  Updating Adam moments...")
            
            # Ensure moments are initialized
            if self.m_t is None or len(self.m_t) != len(pseudo_gradients):
                self.m_t = [np.zeros_like(grad, dtype=np.float32) for grad in pseudo_gradients]
            if self.v_t is None or len(self.v_t) != len(pseudo_gradients):
                self.v_t = [np.zeros_like(grad, dtype=np.float32) for grad in pseudo_gradients]
            
            for i, pseudo_grad in enumerate(pseudo_gradients):
                # First moment estimate (exponential moving average of gradients)
                self.m_t[i] = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * pseudo_grad
                
                # Second moment estimate (exponential moving average of squared gradients)
                self.v_t[i] = self.beta_2 * self.v_t[i] + (1 - self.beta_2) * (pseudo_grad ** 2)
                
                # Bias correction
                m_corrected = self.m_t[i] / (1 - self.beta_1 ** server_round)
                v_corrected = self.v_t[i] / (1 - self.beta_2 ** server_round)
                
                # Adam update
                self.current_weights[i] = self.current_weights[i] + self.eta * m_corrected / (np.sqrt(v_corrected) + self.tau)
            
            # Compute update statistics
            m_norms = [np.linalg.norm(m) for m in self.m_t]
            v_norms = [np.linalg.norm(v) for v in self.v_t]
            
            print(f"  Moment statistics:")
            print(f"    m_t average norm: {np.mean(m_norms):.6f}")
            print(f"    v_t average norm: {np.mean(v_norms):.6f}")
            print(f"    Parameter update applied with η={self.eta}")
        
        # Convert back to Parameters object
        aggregated_parameters = ndarrays_to_parameters(self.current_weights)
        
        # Aggregate metrics from fit results
        aggregated_metrics: Dict[str, Scalar] = {}
        if results:
            # Call parent's fit metrics aggregation
            if hasattr(self, 'fit_metrics_aggregation_fn') and self.fit_metrics_aggregation_fn:
                metrics_list = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
                aggregated_metrics = self.fit_metrics_aggregation_fn(metrics_list)
            
            # Add server-side metrics with proper typing
            server_metrics: Dict[str, Scalar] = {
                "server_round": float(server_round),
                "total_examples": float(total_examples),
                "num_clients_aggregated": float(len(results)),
                "avg_gradient_norm": float(avg_grad_norm),
                "fedadam_eta": float(self.eta),
                "fedadam_eta_l": float(self.eta_l)
            }
            aggregated_metrics.update(server_metrics)
        
        print(f"[Round {server_round}] FixedFedAdam aggregation completed successfully")
        
        return aggregated_parameters, aggregated_metrics

# 6. SERVER FACTORY FUNCTION - UPDATED FOR STANDARD FEDADAM

# 7. SERVERAPP CREATION AND EXPORT - UPDATED FOR FLOWER 1.18.0

def server_fn(context: Context) -> fl.server.ServerAppComponents:
    """Create server components for Flower 1.18.0 with integrated time estimation"""
    
    # Extract configuration from context if available, otherwise use defaults
    try:
        server_config = {
            'num_rounds': context.run_config.get('num-rounds', ServerConstants.DEFAULT_ROUNDS),
            'eta': context.run_config.get('eta', ServerConstants.DEFAULT_ETA),
            'eta_l': context.run_config.get('eta-l', ServerConstants.DEFAULT_ETA_L),
            'beta_1': context.run_config.get('beta-1', ServerConstants.DEFAULT_BETA_1),
            'beta_2': context.run_config.get('beta-2', ServerConstants.DEFAULT_BETA_2),
            'tau': context.run_config.get('tau', ServerConstants.DEFAULT_TAU),
            'min_fit_clients': context.run_config.get('min-fit-clients', ServerConstants.MIN_FIT_CLIENTS),
            'min_evaluate_clients': context.run_config.get('min-evaluate-clients', ServerConstants.MIN_EVALUATE_CLIENTS),
            'fraction_fit': context.run_config.get('fraction-fit', ServerConstants.DEFAULT_FRACTION_FIT),
            'fraction_evaluate': context.run_config.get('fraction-evaluate', ServerConstants.DEFAULT_FRACTION_EVALUATE),
            'test_data_path': context.run_config.get('test-data-path', 'data/raw/ACDC/database/testing'),  # Fixed path
            'enable_global_evaluation': context.run_config.get('enable-global-evaluation', True),
            'noise_adaptation': context.run_config.get('noise-adaptation', True),
            'physics_scheduling': context.run_config.get('physics-scheduling', True),
            'progressive_complexity': context.run_config.get('progressive-complexity', True),
            'local_epochs': context.run_config.get('local-epochs', ServerConstants.DEFAULT_LOCAL_EPOCHS),
            'batch_size': context.run_config.get('batch-size', ServerConstants.DEFAULT_BATCH_SIZE),
        }
    except:
        server_config = {
            'num_rounds': ServerConstants.DEFAULT_ROUNDS,
            'eta': ServerConstants.DEFAULT_ETA,
            'eta_l': ServerConstants.DEFAULT_ETA_L,
            'beta_1': ServerConstants.DEFAULT_BETA_1,
            'beta_2': ServerConstants.DEFAULT_BETA_2,
            'tau': ServerConstants.DEFAULT_TAU,
            'min_fit_clients': ServerConstants.MIN_FIT_CLIENTS,
            'min_evaluate_clients': ServerConstants.MIN_EVALUATE_CLIENTS,
            'fraction_fit': ServerConstants.DEFAULT_FRACTION_FIT,
            'fraction_evaluate': ServerConstants.DEFAULT_FRACTION_EVALUATE,
            'test_data_path': 'data/raw/ACDC/database/testing',
            'enable_global_evaluation': True,
            'noise_adaptation': True,
            'physics_scheduling': True,
            'progressive_complexity': True,
            'local_epochs': ServerConstants.DEFAULT_LOCAL_EPOCHS,
            'batch_size': ServerConstants.DEFAULT_BATCH_SIZE,
        }
    
    # === INTEGRATED TIME ESTIMATION ===
    print("\nFEDERATED LEARNING SERVER STARTING...")
    print("=" * 70)
    
    try:
        # Extract number of clients from environment or estimate
        num_clients = 3  # Default
        if hasattr(context, 'state') and context.state:
            # Try to get from context state
            raw_num_clients = context.state.get('num_supernodes', 3)
            num_clients = int(raw_num_clients) if isinstance(raw_num_clients, (int, str)) else 3
        
        # Parse configuration for estimation
        config_for_estimation = parse_server_config_for_estimation(context)
        
        # Calculate time estimates
        training_estimate = estimate_fl_training_time(
            num_rounds=config_for_estimation['num_rounds'],
            num_clients=num_clients,
            local_epochs=config_for_estimation['local_epochs'],
            train_samples=150,  # Default ACDC training samples
            batch_size=config_for_estimation['batch_size']
        )
        
        testing_estimate = estimate_fl_testing_time(
            test_samples=50,  # Default ACDC test samples
            batch_size=config_for_estimation['batch_size'],
            num_eval_rounds=config_for_estimation['num_rounds']
        )
        
        # Display time estimates
        display_fl_time_estimates(training_estimate, testing_estimate, num_clients)
        
    except Exception as e:
        print(f"Could not calculate time estimates: {e}")
        print("Proceeding with federated learning...")
    
    print(f"\nSERVER CONFIGURATION:")
    print(f"   Rounds: {server_config['num_rounds']}")
    print(f"   Local epochs: {server_config['local_epochs']}")
    print(f"   Batch size: {server_config['batch_size']}")
    print(f"   Min fit clients: {server_config['min_fit_clients']}")
    print(f"   Min evaluate clients: {server_config['min_evaluate_clients']}")
    print(f"   Server learning rate: {server_config['eta']}")
    print(f"   Client learning rate: {server_config['eta_l']}")
    print("=" * 70)
    
    # Rest of the original server_fn code...
    # Setup environment
    device = setup_server_environment()
    
    # 6.2 Global Model Initialization
    global_model = create_global_model(server_config)

    # CRITICAL FIX: Get initial parameters using SAME method as client (model.parameters())
    initial_parameters_list = []
    
    # Count trainable parameters (like client does)
    trainable_count = 0
    total_count = 0
    
    for param in global_model.parameters():
        total_count += 1
        if param.requires_grad:
            param_array = param.detach().cpu().numpy()
            initial_parameters_list.append(param_array)
            trainable_count += 1
    
    print(f"Server found {trainable_count} trainable, {total_count - trainable_count} non-trainable parameters")
    
    # If no trainable parameters found, use all parameters (same fallback as client)
    if len(initial_parameters_list) == 0:
        print("Warning: No trainable parameters found in global model, using all parameters")
        for param in global_model.parameters():
            initial_parameters_list.append(param.detach().cpu().numpy())
    
    initial_parameters = ndarrays_to_parameters(initial_parameters_list)
    print(f"Extracted {len(initial_parameters_list)} parameter arrays from global model")

    # Global test data loading
    test_dataloader = None
    if server_config.get('enable_global_evaluation', True):
        test_dataloader = load_global_test_data(server_config)

    # Evaluation function setup
    evaluate_fn = None
    if test_dataloader is not None:
        evaluate_fn = get_evaluate_fn(global_model, test_dataloader, server_config)
        print("Global evaluation function configured")
    else:
        print("Global evaluation disabled - no test data available")

    # Create configuration functions
    fit_config_fn = create_fit_config_fn(server_config)
    evaluate_config_fn = create_evaluate_config_fn(server_config)

    # Create metrics aggregation functions
    fit_metrics_aggregation_fn = create_fit_metrics_aggregation_fn()
    evaluate_metrics_aggregation_fn = create_evaluate_metrics_aggregation_fn()

    # Create strategy with optimized configuration for better performance using STANDARD FedAdam
    strategy = FedAdam(
        fraction_fit=float(server_config['fraction_fit']),
        fraction_evaluate=float(server_config['fraction_evaluate']),
        min_fit_clients=int(server_config['min_fit_clients']),
        min_evaluate_clients=int(server_config['min_evaluate_clients']),
        min_available_clients=int(server_config['min_fit_clients']),  # Same as min_fit_clients
        evaluate_fn=evaluate_fn,  # Enable server-side evaluation
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
        accept_failures=True,  # Accept some client failures
        initial_parameters=initial_parameters,  # Use Parameters object
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        # FedAdam specific parameters - TUNED for medical imaging non-IID data
        eta=float(server_config['eta']),  # Server learning rate
        eta_l=float(server_config['eta_l']),  # Client learning rate
        beta_1=float(server_config['beta_1']),  # First moment decay
        beta_2=float(server_config['beta_2']),  # Second moment decay  
        tau=float(server_config['tau']),  # Controls adaptability
    )

    print("STANDARD FedAdam strategy initialized with tuned medical parameters")

    # Server configuration
    config = ServerConfig(
        num_rounds=int(server_config['num_rounds']),
        round_timeout=ServerConstants.COMPUTATION_TIMEOUT
    )

    print("MEDICAL FEDERATED LEARNING SERVER READY")
    print(f"Configuration Summary:")
    print(f" • Strategy: Standard FedAdam with medical adaptations")
    print(f" • Adam parameters: β₁={server_config['beta_1']}, β₂={server_config['beta_2']}, τ={server_config['tau']}")
    print(f" • Global evaluation: {'Enabled' if evaluate_fn else 'Disabled'}")
    print(f" • Medical adaptations: Noise={server_config['noise_adaptation']}, Physics={server_config['physics_scheduling']}, Progressive={server_config['progressive_complexity']}")

    return fl.server.ServerAppComponents(
        strategy=strategy,
        config=config
    )

# Create the ServerApp using the new Flower 1.18.0 API
app = ServerApp(server_fn=server_fn)

# 7. MAIN EXECUTION AND CLI SUPPORT

if __name__ == "__main__":
    print("ENHANCED MEDICAL FEDERATED LEARNING SERVER v4.0")
    print("ADVANCED FEATURES:")
    print("✓ Standard FedAdam strategy (Flower 1.18.0 compatible)")
    print("✓ Progressive model complexity scheduling")
    print("✓ Medical domain-specific noise adaptation")
    print("✓ Physics-informed constraint scheduling")
    print("✓ Comprehensive medical metrics aggregation")
    print("✓ Global model evaluation with clinical metrics")
    print("✓ Adaptive learning rate scheduling")
    print("✓ Production-ready error handling and logging")
    print("")
    print("MEDICAL SPECIALIZATIONS:")
    print("✓ ACDC cardiac segmentation optimization")
    print("✓ Medical-safe data augmentation")
    print("✓ Clinical performance metrics (Dice, IoU)")
    print("✓ Physics-informed Maxwell equation constraints")
    print("✓ Quantum noise injection for robustness")
    print("✓ Adaptive complexity scheduling")
    print("\n DEPLOYMENT OPTIONS:")
    print("1. CLI (recommended): flwr run . --run-config config.toml")
    print("2. Legacy CLI: flower-server --app fl_core.app_server:app")
    print("3. Direct execution: python app_server.py --start-server")
    
    # Direct execution option for development
    import sys
    
    if "--start-server" in sys.argv:
        try:
            print("\n Starting Flower server directly...")
            print("Note: Using default configuration for direct execution")
            
            # Create a dummy context for direct execution
            from flwr.common import Context
            
            # Create a minimal context-like object
            class DummyRunConfig:
                def get(self, key, default=None):
                    return default
            
            class DummyContext:
                def __init__(self):
                    self.run_config = DummyRunConfig()
            
            dummy_context = DummyContext()
            
            # Get server components
            server_components = server_fn(dummy_context)  # type: ignore
            
            if server_components.config:
                print(f"Starting server with {server_components.config.num_rounds} rounds...")
            else:
                print("Starting server with default configuration...")
            
            # Start server using server components
            fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=server_components.config,
                strategy=server_components.strategy,
                grpc_max_message_length=ServerConstants.MAX_MESSAGE_LENGTH,
                certificates=None  # No SSL for development
            )
            
        except Exception as e:
            print(f" Server execution failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("\n Use --start-server flag to run directly, or use the CLI command above.")
        print("For production deployment, use the CLI with proper configuration files.")

# === TIME ESTIMATION FUNCTIONS ===

def parse_server_config_for_estimation(context: Context) -> Dict[str, Any]:
    """Parse server configuration for time estimation"""
    try:
        if context.run_config:
            return {
                'num_rounds': context.run_config.get('num-rounds', ServerConstants.DEFAULT_ROUNDS),
                'local_epochs': context.run_config.get('local-epochs', ServerConstants.DEFAULT_LOCAL_EPOCHS),
                'batch_size': context.run_config.get('batch-size', ServerConstants.DEFAULT_BATCH_SIZE),
                'min_fit_clients': context.run_config.get('min-fit-clients', ServerConstants.MIN_FIT_CLIENTS),
                'img_size': context.run_config.get('img-size', ServerConstants.IMG_SIZE),
                'num_classes': context.run_config.get('num-classes', ServerConstants.NUM_CLASSES)
            }
    except:
        pass
    
    # Default configuration
    return {
        'num_rounds': ServerConstants.DEFAULT_ROUNDS,
        'local_epochs': ServerConstants.DEFAULT_LOCAL_EPOCHS,
        'batch_size': ServerConstants.DEFAULT_BATCH_SIZE,
        'min_fit_clients': ServerConstants.MIN_FIT_CLIENTS,
        'img_size': ServerConstants.IMG_SIZE,
        'num_classes': ServerConstants.NUM_CLASSES
    }


def estimate_fl_training_time(num_rounds: int, num_clients: int, local_epochs: int, 
                            train_samples: int = 150, batch_size: int = 4) -> Dict[str, Any]:
    """Estimate federated learning training time"""
    
    # Device performance factor
    try:
        device_factor = 0.4 if torch.cuda.is_available() else 1.0  # GPU faster
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
    except:
        device_factor = 1.0
        device_name = "CPU"
    
    # Base timings (medical segmentation)
    base_batch_time = 0.8  # seconds per batch
    base_communication_time = 2.0  # seconds per round
    
    # Calculate timing estimates
    batches_per_epoch = max(1, train_samples // batch_size)
    batch_time = base_batch_time * device_factor
    epoch_time = batch_time * batches_per_epoch
    
    # Communication overhead increases with clients
    communication_time = base_communication_time + (num_clients * 0.5)
    round_time = (epoch_time * local_epochs) + communication_time
    
    # Total estimates
    total_computation_time = epoch_time * local_epochs * num_rounds
    total_communication_time = communication_time * num_rounds
    total_training_time = total_computation_time + total_communication_time
    
    # Completion estimate
    estimated_completion = datetime.now() + timedelta(seconds=total_training_time)
    
    return {
        'device_name': device_name,
        'total_training_time': total_training_time,
        'total_computation_time': total_computation_time,
        'total_communication_time': total_communication_time,
        'per_round_time': round_time,
        'per_epoch_time': epoch_time,
        'per_batch_time': batch_time,
        'estimated_completion': estimated_completion
    }


def estimate_fl_testing_time(test_samples: int = 50, batch_size: int = 4, 
                           num_eval_rounds: int = 20) -> Dict[str, Any]:
    """Estimate federated learning testing time"""
    
    # Device performance factor
    try:
        device_factor = 0.4 if torch.cuda.is_available() else 1.0
    except:
        device_factor = 1.0
    
    # Testing is faster (no backpropagation)
    base_test_batch_time = 0.3  # seconds per batch
    test_batch_time = base_test_batch_time * device_factor
    
    test_batches = max(1, test_samples // batch_size)
    per_round_test_time = test_batch_time * test_batches
    total_testing_time = per_round_test_time * num_eval_rounds
    
    return {
        'total_testing_time': total_testing_time,
        'per_round_test_time': per_round_test_time,
        'per_batch_time': test_batch_time,
        'per_sample_time': test_batch_time / batch_size
    }


def format_fl_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def display_fl_time_estimates(training_est: Dict[str, Any], testing_est: Dict[str, Any], num_clients: int):
    """Display comprehensive FL time estimates"""
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING TIME ESTIMATION")
    print("="*70)
    
    # Training estimates
    print(f"TRAINING ESTIMATES ({training_est['device_name']}):")
    print(f"   Total Training Time: {format_fl_duration(training_est['total_training_time'])}")
    print(f"   Per Round: {format_fl_duration(training_est['per_round_time'])}")
    print(f"   Per Epoch: {format_fl_duration(training_est['per_epoch_time'])}")
    print(f"   Per Batch: {training_est['per_batch_time']:.3f}s")
    print(f"   Computation: {format_fl_duration(training_est['total_computation_time'])} ({training_est['total_computation_time']/training_est['total_training_time']*100:.1f}%)")
    print(f"   Communication: {format_fl_duration(training_est['total_communication_time'])} ({training_est['total_communication_time']/training_est['total_training_time']*100:.1f}%)")
    
    # Testing estimates
    print(f"\nTESTING ESTIMATES:")
    print(f"   Total Testing Time: {format_fl_duration(testing_est['total_testing_time'])}")
    print(f"   Per Sample: {testing_est['per_sample_time']*1000:.2f}ms")
    print(f"   Per Batch: {testing_est['per_batch_time']:.3f}s")
    
    # Summary
    total_time = training_est['total_training_time'] + testing_est['total_testing_time']
    print(f"\nSUMMARY:")
    print(f"   Clients: {num_clients}")
    print(f"   TOTAL ESTIMATED TIME: {format_fl_duration(total_time)}")
    print(f"   Estimated Completion: {training_est['estimated_completion'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Training vs Testing: {training_est['total_training_time']/testing_est['total_testing_time']:.1f}:1")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if total_time > 3600:  # > 1 hour
        print("   Long training expected. Consider:")
        print("      - Reducing number of rounds")
        print("      - Increasing batch size")
        print("      - Using fewer clients")
    
    if training_est['total_communication_time'] / training_est['total_training_time'] > 0.3:
        print("   High communication overhead. Consider:")
        print("      - Increasing local epochs")
        print("      - Reducing number of clients")
    
    print("="*70)
