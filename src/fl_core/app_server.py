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

# 1.4 Project-Specific Imports - DIRECT IMPORTS ONLY
from src.models.unet_model import (
    RobustMedVFL_UNet,
    CombinedLoss,
    quantum_noise_injection
)
from src.data.data import SimpleMedicalDataset, create_simple_dataloader
from src.utils.seed import set_seed
from src.utils.logger import setup_federated_logger

# 2. GLOBAL CONFIGURATION AND CONSTANTS

class ServerConstants:
    """Centralized server configuration management"""
    
    # AdaFedAdam hyperparameters (MATCHED with config.toml for consistency)
    DEFAULT_ETA = 0.01  # FIXED: Match config.toml (was 0.05)
    DEFAULT_ETA_L = 0.001  # FIXED: Match config.toml (was 0.003) 
    DEFAULT_BETA_1 = 0.9  # FIXED: Match config.toml (was 0.85)
    DEFAULT_BETA_2 = 0.999  # FIXED: Match config.toml (was 0.99)
    DEFAULT_TAU = 1e-7  # Numerical stability
    DEFAULT_ALPHA = 0.5  # FIXED: Match config.toml (was 1.0)
    
    # Aggregation parameters - DYNAMIC BASED ON SUPERNODES
    DEFAULT_FRACTION_FIT = 1.0
    DEFAULT_FRACTION_EVALUATE = 1.0
    MIN_FIT_CLIENTS = 1               # Minimum 1 client
    MIN_EVALUATE_CLIENTS = 1          # Minimum 1 client  
    MIN_AVAILABLE_CLIENTS = 1         # Minimum 1 client
    MAX_CLIENTS_PER_ROUND = 10
    
    # Training constants (MATCHED with config.toml)
    DEFAULT_ROUNDS = 20  # FIXED: Match config.toml (was 25)
    DEFAULT_LOCAL_EPOCHS = 5  # FIXED: Match config.toml (was 3)
    DEFAULT_BATCH_SIZE = 4  # Small batch size for medical segmentation
    
    # Medical imaging constants
    NUM_CLASSES = 4
    IMG_SIZE = 256
    CLASS_NAMES = ['Background', 'RV', 'Myocardium', 'LV']
    
    # Resource limits - INCREASED FOR SIMULATION
    MAX_MEMORY_GB = 16
    COMPUTATION_TIMEOUT = 1200  # 20 minutes (increased from 10)
    ROUND_TIMEOUT = 900         # 15 minutes per round  
    MESSAGE_TIMEOUT = 300       # 5 minutes for messages
    MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 1GB

def setup_server_environment():
    """Setup server environment with optimal configurations"""
    
    # Reduce Flower logging verbosity
    import logging
    flwr_logger = logging.getLogger("flwr")
    flwr_logger.setLevel(logging.WARNING)  # Only show warnings and errors
    
    # Device detection with fallback strategies
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Memory optimization
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # For performance
    
    # Random seed setup using imported function
    set_seed(42)
    
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
    
    # Create the actual model using imported RobustMedVFL_UNet
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
    
    return model

def get_evaluate_fn(
    global_model,
    test_dataloader: Optional[DataLoader],
    config: Dict[str, Any]
) -> Callable:
    """Create evaluation function for global model"""
    
    def evaluate(server_round: int, parameters: Parameters, config_dict: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        try:
            if global_model is None:
                return None
            
            if test_dataloader is None or len(test_dataloader) == 0:
                print(f"[R{server_round}] No test data available - skipping global evaluation")
                return None
            
            # Set parameters to global model using same method as client
            if parameters is not None:
                try:
                    if isinstance(parameters, list):
                        parameters_list = parameters
                    else:
                        parameters_list = parameters_to_ndarrays(parameters)
                    
                    # Use same parameter setting logic as client
                    model_params = list(global_model.parameters())
                    trainable_params = [p for p in model_params if p.requires_grad]
                    
                    # If no trainable parameters found, use all parameters
                    if len(trainable_params) == 0:
                        target_params = model_params
                    else:
                        target_params = trainable_params
                    
                    # Validate parameter count
                    if len(parameters_list) == len(target_params):
                        with torch.no_grad():
                            for param_tensor, new_param_array in zip(target_params, parameters_list):
                                new_param_tensor = torch.from_numpy(new_param_array).to(param_tensor.device)
                                if new_param_tensor.shape == param_tensor.shape:
                                    param_tensor.copy_(new_param_tensor)
                    else:
                        print(f"Warning: Parameter count mismatch: expected {len(target_params)}, got {len(parameters_list)}")
                        
                except Exception as e:
                    print(f"Warning: Could not load parameters: {e}")
            
            # Use new research-grade metrics evaluation
            from src.utils.metrics import evaluate_model_with_research_metrics, convert_metrics_for_fl_server
            
            device = next(global_model.parameters()).device
            
            # Get comprehensive research metrics
            research_metrics = evaluate_model_with_research_metrics(
                model=global_model,
                dataloader=test_dataloader,
                device=device,
                num_classes=4,
                class_names=['Background', 'RV', 'Myocardium', 'LV'],
                return_detailed=False
            )
            
            # Convert to FL-compatible format
            fl_metrics = convert_metrics_for_fl_server(research_metrics)
            
            # Add server-specific information with proper typing
            server_info: Dict[str, Scalar] = {
                "server_round": float(server_round),
                "global_evaluation": 1.0,
                "model_name": "RobustMedVFL_UNet"
            }
            
            # Merge metrics dictionaries
            final_metrics: Dict[str, Scalar] = {}
            for key, value in fl_metrics.items():
                final_metrics[key] = float(value) if isinstance(value, (int, float)) else value
            for key, value in server_info.items():
                final_metrics[key] = value
            
            # Use eval_loss as the primary loss metric for FL optimization
            primary_loss = float(fl_metrics.get('eval_loss', 1.0))
            
            # Extract key metrics for consistent access
            mean_dice = research_metrics.get('mean_dice', 0.0)
            total_samples = research_metrics.get('total_samples', 0)
            
            # Print research summary for monitoring
            if server_round % 5 == 0:  # Every 5 rounds
                from src.utils.metrics import print_research_metrics_summary
                print_research_metrics_summary(research_metrics, f"GLOBAL EVALUATION - Round {server_round}")
            else:
                # COMPACT CUSTOM FORMAT - Override Flower's verbose logging
                mean_dice_fg = research_metrics.get('mean_dice_fg', 0.0)
                mean_iou = research_metrics.get('mean_iou', 0.0)
                
                # Per-class dice for detailed monitoring
                dice_scores = research_metrics.get('dice_scores', [0,0,0,0])
                dice_bg, dice_rv, dice_myo, dice_lv = dice_scores[:4] if len(dice_scores) >= 4 else [0,0,0,0]
                
                # SINGLE LINE COMPACT FORMAT
                print(f"[R{server_round:02d}] Loss={primary_loss:.4f} | Dice={mean_dice:.4f} | IoU={mean_iou:.4f} | FG={mean_dice_fg:.4f} | BG={dice_bg:.3f} RV={dice_rv:.3f} Myo={dice_myo:.3f} LV={dice_lv:.3f} | Samples={total_samples}")
            
            # Create minimal metrics to prevent verbose Flower logging
            minimal_metrics = {
                "server_round": float(server_round),
                "eval_loss": primary_loss,
                "dice_score": mean_dice,
                "samples": float(total_samples)
            }
            
            return primary_loss, minimal_metrics
            
        except Exception as e:
            print(f"Global evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return evaluate

def load_global_test_data(config: Dict[str, Any]) -> Optional[DataLoader]:
    """Load global test dataset for server-side evaluation"""
    
    # Load test data from configured path
    test_data_path = config.get('test_data_path', 'data/raw/ACDC/database/testing')
    
    if os.path.exists(test_data_path):
        test_dataloader = create_simple_dataloader(
            data_dir=test_data_path,
            batch_size=config.get('test_batch_size', 4),
            shuffle=False,
            augment=False,  # FIXED: No augmentation for testing!
            num_workers=0
        )
        
        return test_dataloader
        
    else:
        # Try alternative paths
        alt_paths = [
            'data/raw/ACDC/database/training',  # Use training as fallback
            'data/raw/ACDC/training',
            'data/raw/ACDC'
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                test_dataloader = create_simple_dataloader(
                    data_dir=alt_path,
                    batch_size=config.get('test_batch_size', 4),
                    shuffle=False,
                    augment=False,  # FIXED: No augmentation for testing!
                    num_workers=0
                )
                return test_dataloader
        
        return None

# 5. ADAPTIVE FEDERATED ADAM IMPLEMENTATION

class AdaFedAdam(FedAdam):
    """
    Adaptive Federated Adam (AdaFedAdam) with fairness control.
    
    Based on: "AdaFedAdam: Adaptive Federated Optimization with Fairness" (arXiv:2301.09357)
    
    Key improvements over FedAdam:
    - Adaptive fairness weighting based on client loss ratios
    - Better handling of non-IID data distribution
    - Fairness control through alpha hyperparameter
    """

    def __init__(
        self,
        *,
        alpha: float = 2.0,  # Fairness hyperparameter
        initial_parameters: Parameters,
        **kwargs
    ):
        """
        Initialize AdaFedAdam strategy.
        
        Args:
            alpha: Fairness hyperparameter controlling adaptive weighting
            initial_parameters: Initial model parameters
            **kwargs: Additional arguments passed to FedAdam
        """
        super().__init__(initial_parameters=initial_parameters, **kwargs)
        self.alpha = alpha
        self.client_initial_losses = {}  # Store initial loss for each client
        self.client_latest_losses = {}   # Store latest loss for each client
        
        print(f"✓ AdaFedAdam initialized: α={alpha}, Medical FL with fairness weighting")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates with adaptive fairness weighting.
        
        Args:
            server_round: Current federated learning round
            results: Client fit results
            failures: Failed client results
            
        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            print(f"[R{server_round}] No results to aggregate")
            return None, {}

        # Step 1: Gather client losses and weights (silent processing)
        client_ids = []
        client_losses = []
        client_updates = []
        num_examples_list = []
        
        # Step 2: Process each client result silently
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            
            # Get train_loss from metrics (required for AdaFedAdam)
            train_loss = fit_res.metrics.get("train_loss", 1.0)
            if train_loss == 0.0:  # Avoid division by zero
                train_loss = 1e-6
                
            self.client_latest_losses[cid] = train_loss
            
            # Store initial loss on first round
            if cid not in self.client_initial_losses:
                self.client_initial_losses[cid] = train_loss
            
            client_ids.append(cid)
            client_losses.append(train_loss)
            client_updates.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples_list.append(fit_res.num_examples)

        # Step 3: Compute adaptive weights (AdaFedAdam Eq.9)
        weights = []
        total_examples = sum(num_examples_list)
        
        for cid, loss, num_examples in zip(client_ids, client_losses, num_examples_list):
            initial_loss = self.client_initial_losses.get(cid, loss)
            
            # Compute loss improvement ratio
            if loss > 0:
                improvement_ratio = initial_loss / loss
            else:
                improvement_ratio = 1.0
            
            # Adaptive weight computation (combines fairness and data size)
            fairness_weight = improvement_ratio ** self.alpha
            data_weight = num_examples / total_examples
            
            # Combine fairness and data weighting
            adaptive_weight = fairness_weight * data_weight
            weights.append(adaptive_weight)

        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        else:
            # Fallback to uniform weighting
            weights = [1.0 / len(client_ids) for _ in client_ids]

        # Step 4: Weighted average of updates (silent processing)
        agg_update = []
        
        if len(client_updates[0]) > 0:
            for i in range(len(client_updates[0])):
                weighted_sum = sum(w * update[i] for w, update in zip(weights, client_updates))
                agg_update.append(weighted_sum)
        else:
            return None, {}

        # Step 5: Adam moment updates (same as FedAdam but with proper initialization)
        if not hasattr(self, "current_weights") or self.current_weights is None:
            self.current_weights = [np.copy(w) for w in agg_update]
            self.m_t = [np.zeros_like(w, dtype=np.float32) for w in agg_update]
            self.v_t = [np.zeros_like(w, dtype=np.float32) for w in agg_update]
        else:
            # Ensure m_t and v_t are initialized
            if not hasattr(self, "m_t") or self.m_t is None:
                self.m_t = [np.zeros_like(w, dtype=np.float32) for w in agg_update]
            if not hasattr(self, "v_t") or self.v_t is None:
                self.v_t = [np.zeros_like(w, dtype=np.float32) for w in agg_update]
            
            # Compute parameter updates (pseudo-gradients)
            delta_t = [agg - curr for agg, curr in zip(agg_update, self.current_weights)]
            
            # Adam moment updates - with proper type checking
            if self.m_t is not None and self.v_t is not None:
                self.m_t = [
                    self.beta_1 * m + (1 - self.beta_1) * d
                    for m, d in zip(self.m_t, delta_t)
                ]
                self.v_t = [
                    self.beta_2 * v + (1 - self.beta_2) * (d ** 2)
                    for v, d in zip(self.v_t, delta_t)
                ]
                
                # Bias correction
                m_hat = [m / (1 - self.beta_1 ** server_round) for m in self.m_t]
                v_hat = [v / (1 - self.beta_2 ** server_round) for v in self.v_t]
                
                # Adaptive learning rate with bias correction
                eta_norm = (
                    self.eta
                    * np.sqrt(1 - self.beta_2 ** server_round)
                    / (1 - self.beta_1 ** server_round)
                )
                
                # Parameter update
                self.current_weights = [
                    curr + eta_norm * m / (np.sqrt(v) + self.tau)
                    for curr, m, v in zip(self.current_weights, m_hat, v_hat)
                ]

        # Step 6: Return new parameters and aggregated metrics
        new_parameters = ndarrays_to_parameters(self.current_weights)
        
        # Aggregate fit metrics using parent's method if available
        aggregated_metrics: Dict[str, Scalar] = {}
        if hasattr(self, 'fit_metrics_aggregation_fn') and self.fit_metrics_aggregation_fn:
            metrics_list = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(metrics_list)
        
        # Add AdaFedAdam specific metrics
        avg_loss = float(np.mean(client_losses))
        weight_entropy = float(self._compute_weight_entropy(weights))
        fairness_variance = float(np.var([w * len(weights) for w in weights]))
        
        adafedadam_metrics: Dict[str, Scalar] = {
            "adafedadam_alpha": float(self.alpha),
            "server_round": float(server_round),
            "num_clients_aggregated": float(len(results)),
            "total_examples": float(sum(num_examples_list)),
            "avg_client_loss": avg_loss,
            "weight_entropy": weight_entropy,
            "fairness_variance": fairness_variance,
        }
        
        # Merge metrics
        aggregated_metrics.update(adafedadam_metrics)
        
        # OPTIMIZED: Single compact research metrics line
        loss_improvements = [
            (self.client_initial_losses.get(cid, loss) - loss) / max(self.client_initial_losses.get(cid, loss), 1e-6)
            for cid, loss in zip(client_ids, client_losses)
        ]
        avg_improvement = np.mean(loss_improvements) if loss_improvements else 0.0
        client_weights_summary = f"[{min(weights):.2f}→{max(weights):.2f}]"
        
        # COMPACT ADAFEDADAM FORMAT
        print(f"[R{server_round:02d}] AdaFedAdam: {len(results)}C | Loss={avg_loss:.4f} | Improve={avg_improvement:.1%} | Weights={client_weights_summary}")
        
        return new_parameters, aggregated_metrics
    
    def _compute_weight_entropy(self, weights: List[float]) -> float:
        """Compute entropy of aggregation weights to measure fairness."""
        weights_array = np.array(weights, dtype=np.float32)
        weights_array = weights_array / np.sum(weights_array)  # Ensure normalized
        weights_array = weights_array[weights_array > 0]  # Remove zero weights
        
        if len(weights_array) <= 1:
            return 0.0
        
        entropy = -np.sum(weights_array * np.log(weights_array + 1e-8))
        max_entropy = np.log(len(weights_array))  # Maximum possible entropy
        
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0  # Normalized entropy

# 6. SERVER FACTORY FUNCTION - UPDATED FOR STANDARD FEDADAM

# 7. SERVERAPP CREATION AND EXPORT - UPDATED FOR FLOWER 1.18.0

def server_fn(context: Context) -> fl.server.ServerAppComponents:
    """Create server components for Flower 1.18.0 with AdaFedAdam strategy"""
    
    # Extract configuration from context if available, otherwise use defaults
    try:
        server_config = {
            'num_rounds': context.run_config.get('num-rounds', ServerConstants.DEFAULT_ROUNDS),
            'eta': context.run_config.get('eta', ServerConstants.DEFAULT_ETA),
            'eta_l': context.run_config.get('eta-l', ServerConstants.DEFAULT_ETA_L),
            'beta_1': context.run_config.get('beta-1', ServerConstants.DEFAULT_BETA_1),
            'beta_2': context.run_config.get('beta-2', ServerConstants.DEFAULT_BETA_2),
            'tau': context.run_config.get('tau', ServerConstants.DEFAULT_TAU),
            'alpha': context.run_config.get('alpha', ServerConstants.DEFAULT_ALPHA),  # AdaFedAdam fairness parameter
            'min_fit_clients': context.run_config.get('min-fit-clients', ServerConstants.MIN_FIT_CLIENTS),
            'min_evaluate_clients': context.run_config.get('min-evaluate-clients', ServerConstants.MIN_EVALUATE_CLIENTS),
            'fraction_fit': context.run_config.get('fraction-fit', ServerConstants.DEFAULT_FRACTION_FIT),
            'fraction_evaluate': context.run_config.get('fraction-evaluate', ServerConstants.DEFAULT_FRACTION_EVALUATE),
            'test_data_path': context.run_config.get('test-data-path', 'data/raw/ACDC/database/testing'),
            'enable_global_evaluation': context.run_config.get('enable-global-evaluation', True),
            'noise_adaptation': context.run_config.get('noise-adaptation', True),
            'physics_scheduling': context.run_config.get('physics-scheduling', True),
            'progressive_complexity': context.run_config.get('progressive-complexity', True),
            'local_epochs': context.run_config.get('local-epochs', ServerConstants.DEFAULT_LOCAL_EPOCHS),
            'batch_size': context.run_config.get('batch-size', ServerConstants.DEFAULT_BATCH_SIZE),
            # Timeout configurations
            'round_timeout': context.run_config.get('round-timeout', ServerConstants.ROUND_TIMEOUT),
            'message_timeout': context.run_config.get('message-timeout', ServerConstants.MESSAGE_TIMEOUT),
            'metadata_ttl': context.run_config.get('metadata-ttl', 3600),
        }
    except:
        server_config = {
            'num_rounds': ServerConstants.DEFAULT_ROUNDS,
            'eta': ServerConstants.DEFAULT_ETA,
            'eta_l': ServerConstants.DEFAULT_ETA_L,
            'beta_1': ServerConstants.DEFAULT_BETA_1,
            'beta_2': ServerConstants.DEFAULT_BETA_2,
            'tau': ServerConstants.DEFAULT_TAU,
            'alpha': ServerConstants.DEFAULT_ALPHA,
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
            # Timeout configurations
            'round_timeout': ServerConstants.ROUND_TIMEOUT,
            'message_timeout': ServerConstants.MESSAGE_TIMEOUT,
            'metadata_ttl': 3600,
        }
    
    # === INTEGRATED TIME ESTIMATION ===
    print("AdaFedAdam Server Starting...")
    
    try:
        # Extract number of clients dynamically
        num_clients = None
        
        # Try to get from context run_config
        if hasattr(context, 'run_config') and context.run_config:
            raw_num_clients = getattr(context.run_config, 'num_supernodes', None)
            if raw_num_clients:
                try:
                    num_clients = int(str(raw_num_clients))
                except (ValueError, TypeError):
                    pass
        
        # Try to get from context state if available
        if not num_clients and hasattr(context, 'state') and context.state:
            raw_num_clients = context.state.get('num_supernodes')
            if raw_num_clients:
                try:
                    num_clients = int(str(raw_num_clients))
                except (ValueError, TypeError):
                    pass
        
        # Check environment variables for supernodes
        if not num_clients:
            env_vars_to_check = ['FLWR_NUM_SUPERNODES', 'NUM_SUPERNODES', 'FLOWER_NUM_SUPERNODES']
            for env_var in env_vars_to_check:
                env_value = os.environ.get(env_var)
                if env_value:
                    try:
                        num_clients = int(env_value)
                        break
                    except ValueError:
                        continue
        
        # Try to parse from sys.argv
        if not num_clients:
            import sys
            try:
                argv = sys.argv
                for i, arg in enumerate(argv):
                    if arg == '--num-supernodes' and i + 1 < len(argv):
                        num_clients = int(argv[i + 1])
                        break
            except (ValueError, IndexError):
                pass
        
        # Use default if not found
        if num_clients is None:
            num_clients = 5
        
        # Calculate time estimates
        config_for_estimation = parse_server_config_for_estimation(context)
        training_estimate = estimate_fl_training_time(
            num_rounds=config_for_estimation['num_rounds'],
            num_clients=num_clients,
            local_epochs=config_for_estimation['local_epochs'],
            train_samples=150,
            batch_size=config_for_estimation['batch_size']
        )
        
        testing_estimate = estimate_fl_testing_time(
            test_samples=50,
            batch_size=config_for_estimation['batch_size'],
            num_eval_rounds=config_for_estimation['num_rounds']
        )
        
        # Display concise time estimates
        total_time = training_estimate['total_training_time'] + testing_estimate['total_testing_time']
        print(f"Estimated completion: {format_fl_duration(total_time)} | "
              f"Clients: {num_clients} | Device: {training_estimate['device_name']}")
        
    except Exception:
        pass  # Skip time estimation if it fails
    
    # Setup environment
    device = setup_server_environment()
    
    # Global Model Initialization
    global_model = create_global_model(server_config)

    # Get initial parameters using SAME method as client (model.parameters())
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
    
    # If no trainable parameters found, use all parameters (same fallback as client)
    if len(initial_parameters_list) == 0:
        for param in global_model.parameters():
            initial_parameters_list.append(param.detach().cpu().numpy())
    
    initial_parameters = ndarrays_to_parameters(initial_parameters_list)

    # Global test data loading
    test_dataloader = None
    if server_config.get('enable_global_evaluation', True):
        test_dataloader = load_global_test_data(server_config)

    # Evaluation function setup
    evaluate_fn = None
    if test_dataloader is not None:
        evaluate_fn = get_evaluate_fn(global_model, test_dataloader, server_config)

    # Create configuration functions
    fit_config_fn = create_fit_config_fn(server_config)
    evaluate_config_fn = create_evaluate_config_fn(server_config)

    # Create metrics aggregation functions
    fit_metrics_aggregation_fn = create_fit_metrics_aggregation_fn()
    evaluate_metrics_aggregation_fn = create_evaluate_metrics_aggregation_fn()

    # Create AdaFedAdam strategy with optimized configuration for medical imaging
    strategy = AdaFedAdam(
        alpha=float(server_config['alpha']),  # Fairness hyperparameter
        initial_parameters=initial_parameters,
        fraction_fit=float(server_config['fraction_fit']),
        fraction_evaluate=float(server_config['fraction_evaluate']),
        min_fit_clients=int(server_config['min_fit_clients']),
        min_evaluate_clients=int(server_config['min_evaluate_clients']),
        min_available_clients=int(server_config['min_fit_clients']),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
        accept_failures=True,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        # AdaFedAdam/FedAdam specific parameters - TUNED for medical imaging
        eta=float(server_config['eta']),  # Server learning rate
        eta_l=float(server_config['eta_l']),  # Client learning rate
        beta_1=float(server_config['beta_1']),  # First moment decay
        beta_2=float(server_config['beta_2']),  # Second moment decay  
        tau=float(server_config['tau']),  # Numerical stability
    )

    # Server configuration
    config = ServerConfig(
        num_rounds=int(server_config['num_rounds']),
        round_timeout=int(server_config.get('round_timeout', ServerConstants.ROUND_TIMEOUT))
    )

    print(f"AdaFedAdam Server Ready: {server_config['num_rounds']} rounds | "
          f"α={server_config['alpha']} | "
          f"Eval={'On' if evaluate_fn else 'Off'}")

    return fl.server.ServerAppComponents(
        strategy=strategy,
        config=config
    )

# Create the ServerApp using the new Flower 1.18.0 API
app = ServerApp(server_fn=server_fn)

# 7. MAIN EXECUTION AND CLI SUPPORT

if __name__ == "__main__":
    print("Medical Federated Learning Server v4.0")
    print("✓ AdaFedAdam strategy (Flower 1.18.0)")
    print("✓ Medical domain optimizations")
    print("✓ Adaptive fairness weighting")
    print("\nDeployment: flwr run . --run-config config.toml")
    
    # Direct execution option for development
    import sys
    
    if "--start-server" in sys.argv:
        try:
            print("Starting Flower server...")
            
            # Create a dummy context for direct execution
            from flwr.common import Context
            
            class DummyRunConfig:
                def get(self, key, default=None):
                    return default
            
            class DummyContext:
                def __init__(self):
                    self.run_config = DummyRunConfig()
            
            dummy_context = DummyContext()
            server_components = server_fn(dummy_context)  # type: ignore
            
            fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=server_components.config,
                strategy=server_components.strategy,
                grpc_max_message_length=ServerConstants.MAX_MESSAGE_LENGTH,
                certificates=None
            )
            
        except Exception as e:
            print(f"Server execution failed: {type(e).__name__}: {str(e)}")
    else:
        print("Use --start-server flag to run directly")

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
