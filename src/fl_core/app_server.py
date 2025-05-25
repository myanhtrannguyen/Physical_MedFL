"""
Enhanced Federated Learning Server for Medical Image Segmentation
Comprehensive implementation with adaptive strategies and medical domain specialization
"""

# =============================================================================
# 1. HEADER AND IMPORTS SECTION
# =============================================================================

# 1.1 Standard Library Imports
import os
import sys
import logging
import warnings
import time
import json
import pickle
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import OrderedDict
from pathlib import Path

# 1.2 Scientific Computing Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# 1.3 Flower Framework Imports
import flwr as fl
from flwr.server import ServerApp, ServerConfig
try:
    from flwr.server import ServerAppComponents
except ImportError:
    # Fallback for older Flower versions
    from typing import NamedTuple
    class ServerAppComponents(NamedTuple):
        strategy: Strategy
        config: ServerConfig
from flwr.server.strategy import Strategy
from flwr.common import Context, Parameters, FitRes, EvaluateRes, Scalar
from flwr.common import FitIns, EvaluateIns, GetParametersIns
from flwr.server.client_proxy import ClientProxy

# 1.4 Project-Specific Imports
# Updated imports to use the refactored structure
from ..models.mlp_model import (
    OptimizedRobustMedVFL_UNet, 
    OptimizedCombinedLoss, 
    optimized_quantum_noise_injection
)
from ..data.dataset import ACDCDataset
from ..data.loader import create_acdc_dataloader, create_dataloader
from ..utils.seed import set_seed
from ..utils.logger import setup_federated_logger
from ..utils.metrics import evaluate_metrics, compute_class_weights, print_metrics_summary

# =============================================================================
# 2. GLOBAL CONFIGURATION AND CONSTANTS
# =============================================================================

# 2.1 Server Configuration Constants  
class ServerConstants:
    """Centralized server configuration management"""
    
    # Default hyperparameters
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_BETA1 = 0.9
    DEFAULT_BETA2 = 0.999
    DEFAULT_EPSILON = 1e-8
    DEFAULT_WEIGHT_DECAY = 1e-5
    
    # Aggregation parameters
    DEFAULT_FRACTION_FIT = 1.0
    DEFAULT_FRACTION_EVALUATE = 1.0
    MIN_FIT_CLIENTS = 1
    MIN_EVALUATE_CLIENTS = 1
    MIN_AVAILABLE_CLIENTS = 1
    MAX_CLIENTS_PER_ROUND = 10
    
    # Training constants
    DEFAULT_ROUNDS = 10
    DEFAULT_LOCAL_EPOCHS = 5
    DEFAULT_BATCH_SIZE = 8
    
    # Medical imaging constants
    NUM_CLASSES = 4
    IMG_SIZE = 256
    CLASS_NAMES = ['Background', 'RV', 'Myocardium', 'LV']
    
    # Resource limits
    MAX_MEMORY_GB = 16
    COMPUTATION_TIMEOUT = 600  # 10 minutes
    MAX_MESSAGE_LENGTH = 512 * 1024 * 1024  # 512MB

# 2.2 Environment Setup
def setup_server_environment():
    """Setup server environment with optimal configurations"""
    # Device detection with fallback strategies
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU - consider GPU for better performance")
    
    # Memory optimization
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # For performance
    
    # Random seed management
    set_seed(42)
    
    return device

# Global device setup
DEVICE = setup_server_environment()

# =============================================================================
# 3. MEDICAL FEDADAM STRATEGY CLASS
# =============================================================================

class MedicalFedAdam(Strategy):
    """
    Advanced Federated Adam strategy specialized for medical image segmentation
    Implements adaptive model complexity, medical domain constraints, and robust aggregation
    """
    
    def __init__(
        self,
        # 3.1.1 Core Parameters
        learning_rate: float = ServerConstants.DEFAULT_LEARNING_RATE,
        beta1: float = ServerConstants.DEFAULT_BETA1,
        beta2: float = ServerConstants.DEFAULT_BETA2,
        epsilon: float = ServerConstants.DEFAULT_EPSILON,
        weight_decay: float = ServerConstants.DEFAULT_WEIGHT_DECAY,
        
        # Client selection parameters
        fraction_fit: float = ServerConstants.DEFAULT_FRACTION_FIT,
        fraction_evaluate: float = ServerConstants.DEFAULT_FRACTION_EVALUATE,
        min_fit_clients: int = ServerConstants.MIN_FIT_CLIENTS,
        min_evaluate_clients: int = ServerConstants.MIN_EVALUATE_CLIENTS,
        min_available_clients: int = ServerConstants.MIN_AVAILABLE_CLIENTS,
        max_clients_per_round: int = ServerConstants.MAX_CLIENTS_PER_ROUND,
        
        # Medical-specific parameters
        noise_adaptation: bool = True,
        physics_scheduling: bool = True,
        progressive_complexity: bool = True,
        
        # Evaluation function
        evaluate_fn: Optional[Callable] = None,
        
        # Additional parameters
        num_rounds: int = ServerConstants.DEFAULT_ROUNDS,
    ):
        """Initialize MedicalFedAdam strategy with comprehensive configuration"""
        
        # 3.1.2 Adam State Management
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # Moment estimates (initialized when first parameters received)
        self.m: Optional[List[torch.Tensor]] = None  # First moment
        self.v: Optional[List[torch.Tensor]] = None  # Second moment
        self.t = 0  # Time step for bias correction
        
        # Global model state
        self.current_global_params: Optional[Parameters] = None
        self.convergence_history = []
        self.loss_history = []
        
        # Client selection parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.max_clients_per_round = max_clients_per_round
        
        # 3.1.3 Medical Domain Configuration
        self.noise_adaptation = noise_adaptation
        self.physics_scheduling = physics_scheduling
        self.progressive_complexity = progressive_complexity
        self.num_rounds = num_rounds
        
        # Evaluation function
        self.evaluate_fn = evaluate_fn
        
        # Performance tracking
        self.round_metrics = {}
        self.best_global_metric = 0.0
        self.rounds_without_improvement = 0
        self.max_patience = 3
        
        # Client quality tracking
        self.client_quality_scores = {}
        self.client_reliability_scores = {}
        
        # Setup logging
        self.logger = setup_federated_logger(client_id=None, log_dir="logs/server")
        self.logger.info("MedicalFedAdam strategy initialized")
        self.logger.info(f"Configuration: lr={learning_rate}, rounds={num_rounds}")
        
    def _get_pruning_config(self, server_round: int) -> Dict[str, Any]:
        """
        Get adaptive pruning configuration based on round number
        Implements progressive complexity scheduling
        """
        if not self.progressive_complexity:
            # Use balanced configuration throughout
            return {
                'noise_processing_levels': [0, 1],
                'maxwell_solver_levels': [0, 1],
                'dropout_positions': [0],
                'skip_quantum_noise': False
            }
        
        # Progressive complexity scheduling
        if server_round <= 5:
            # Early rounds: Simplified model
            config = {
                'noise_processing_levels': [0],  # Only first level
                'maxwell_solver_levels': [],     # No physics initially
                'dropout_positions': [0],
                'skip_quantum_noise': True       # No quantum noise initially
            }
            self.logger.info(f"Round {server_round}: Using simplified model configuration")
            
        elif server_round <= 15:
            # Middle rounds: Progressive complexity
            config = {
                'noise_processing_levels': [0, 1],
                'maxwell_solver_levels': [0],    # Limited physics
                'dropout_positions': [0],
                'skip_quantum_noise': False
            }
            self.logger.info(f"Round {server_round}: Using moderate complexity configuration")
            
        else:
            # Later rounds: Full complexity
            config = {
                'noise_processing_levels': [0, 1, 2],
                'maxwell_solver_levels': [0, 1],
                'dropout_positions': [0],
                'skip_quantum_noise': False
            }
            self.logger.info(f"Round {server_round}: Using full complexity configuration")
            
        return config
    
    def _get_noise_schedule(self, server_round: int) -> Dict[str, float]:
        """Get adaptive noise scheduling parameters"""
        if not self.noise_adaptation:
            return {
                'quantum_noise_factor': 0.05,
                'dropout_rate': 0.1,
                'augmentation_intensity': 0.5
            }
        
        # Progressive noise scheduling
        progress = min(server_round / self.num_rounds, 1.0)
        
        # Quantum noise: Start low, increase gradually
        quantum_noise_factor = 0.02 + 0.03 * progress
        
        # Dropout: Adaptive based on convergence
        if len(self.loss_history) > 2:
            recent_improvement = self.loss_history[-2] - self.loss_history[-1]
            if recent_improvement < 0.01:  # Poor improvement
                dropout_rate = min(0.2, 0.1 + 0.05 * (1 - recent_improvement))
            else:
                dropout_rate = 0.1
        else:
            dropout_rate = 0.1
        
        # Augmentation: Medical-safe progression
        augmentation_intensity = 0.3 + 0.2 * progress
        
        return {
            'quantum_noise_factor': quantum_noise_factor,
            'dropout_rate': dropout_rate,
            'augmentation_intensity': augmentation_intensity
        }
    
    def _get_learning_rate_schedule(self, server_round: int) -> float:
        """Get adaptive learning rate based on round and convergence"""
        base_lr = self.learning_rate
        
        # Exponential decay with step scheduling
        if server_round <= 5:
            lr_multiplier = 1.0
        elif server_round <= 10:
            lr_multiplier = 0.5
        elif server_round <= 15:
            lr_multiplier = 0.2
        else:
            lr_multiplier = 0.1
        
        # Adaptive adjustment based on convergence
        if len(self.loss_history) > 3:
            recent_losses = self.loss_history[-3:]
            if all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                # Loss is increasing - reduce learning rate
                lr_multiplier *= 0.5
                self.logger.warning(f"Loss increasing - reducing learning rate to {base_lr * lr_multiplier}")
        
        return base_lr * lr_multiplier
    
    # 3.2 Client Configuration Methods
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure clients for training with adaptive parameters
        Implements round-based adaptation and medical domain optimization
        """
        self.logger.info(f"{'='*60}")
        self.logger.info(f"ROUND {server_round} - CONFIGURE FIT")
        self.logger.info(f"{'='*60}")
        
        # Get adaptive configurations
        pruning_config = self._get_pruning_config(server_round)
        noise_schedule = self._get_noise_schedule(server_round)
        learning_rate = self._get_learning_rate_schedule(server_round)
        
        # Adaptive epoch scheduling
        if server_round <= 3:
            local_epochs = 6  # More epochs early for better learning
        elif server_round <= 8:
            local_epochs = 5  # Moderate epochs
        else:
            local_epochs = 4  # Fewer epochs for fine-tuning
        
        # Create comprehensive configuration
        config: Dict[str, Scalar] = {
            # Training parameters
            "server_round": server_round,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "weight_decay": self.weight_decay,
                         "batch_size": ServerConstants.DEFAULT_BATCH_SIZE,
             
             # Model parameters
             "dropout_rate": noise_schedule['dropout_rate'],
             "quantum_noise_factor": noise_schedule['quantum_noise_factor'],
             
             # Pruning configuration (serialized as JSON)
             "pruning_config": json.dumps(pruning_config),
             
             # Medical parameters
             "augmentation_intensity": noise_schedule['augmentation_intensity'],
             "num_classes": ServerConstants.NUM_CLASSES,
            
            # Round metadata
            "total_rounds": self.num_rounds,
            "convergence_status": "improving" if self.rounds_without_improvement < 2 else "stable",
            
            # Physics scheduling
            "enable_physics": self.physics_scheduling and server_round > 5,
        }
        
        # Log configuration
        self.logger.info(f"Training configuration for round {server_round}:")
        self.logger.info(f"  Local epochs: {local_epochs}")
        self.logger.info(f"  Learning rate: {learning_rate:.2e}")
        self.logger.info(f"  Dropout rate: {noise_schedule['dropout_rate']:.3f}")
        self.logger.info(f"  Quantum noise: {noise_schedule['quantum_noise_factor']:.3f}")
        self.logger.info(f"  Physics enabled: {config['enable_physics']}")
        
                 # Client selection with quality awareness
         available_clients_dict = client_manager.all()
         available_clients = list(available_clients_dict.values()) if isinstance(available_clients_dict, dict) else available_clients_dict
         
         # Resource-aware and quality-based selection
         selected_clients = self._select_clients_intelligently(
             available_clients, 
             server_round,
             min(self.max_clients_per_round, len(available_clients))
         )
        
        self.logger.info(f"  Selected {len(selected_clients)} clients for training")
        
        # Create FitIns for each client
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in selected_clients]
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure clients for evaluation with medical-specific metrics"""
        self.logger.info(f"CONFIGURE EVALUATION - Round {server_round}")
        
        # Evaluation configuration
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "total_rounds": self.num_rounds,
            "evaluation_depth": "comprehensive" if server_round % 5 == 0 else "standard",
            "num_classes": ServerConfig.NUM_CLASSES,
            "collect_medical_metrics": True,
            "collect_physics_metrics": self.physics_scheduling,
        }
        
        # Select clients for evaluation
        available_clients = client_manager.all()
        num_eval_clients = min(
            max(self.min_evaluate_clients, len(available_clients) // 2),
            len(available_clients)
        )
        
        selected_clients = client_manager.sample(
            num_clients=num_eval_clients,
            min_num_clients=self.min_evaluate_clients
        )
        
        self.logger.info(f"  Selected {len(selected_clients)} clients for evaluation")
        
        eval_ins = EvaluateIns(parameters, config)
        return [(client, eval_ins) for client in selected_clients]
    
    def _select_clients_intelligently(
        self, 
        available_clients: List[ClientProxy], 
        server_round: int,
        num_clients: int
    ) -> List[ClientProxy]:
        """
        Intelligent client selection based on quality, reliability, and diversity
        """
        if len(available_clients) <= num_clients:
            return available_clients
        
        # Simple selection for early rounds when we don't have history
        if server_round <= 2 or not self.client_quality_scores:
            return available_clients[:num_clients]
        
        # Score-based selection for later rounds
        client_scores = []
        for client in available_clients:
            cid = client.cid
            
            # Base score
            quality_score = self.client_quality_scores.get(cid, 0.5)
            reliability_score = self.client_reliability_scores.get(cid, 0.5)
            
            # Combined score with weights
            combined_score = 0.6 * quality_score + 0.4 * reliability_score
            
            client_scores.append((client, combined_score))
        
        # Sort by score and select top clients
        client_scores.sort(key=lambda x: x[1], reverse=True)
        selected_clients = [client for client, _ in client_scores[:num_clients]]
        
        self.logger.info(f"  Client selection based on quality scores")
        return selected_clients
    
    # 3.3 Aggregation Methods
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Advanced parameter aggregation using FedAdam with medical domain adaptations
        """
        self.logger.info(f"AGGREGATE FIT RESULTS - Round {server_round}")
        
        # 3.3.1 Pre-aggregation validation
        if failures:
            self.logger.warning(f"Training failures: {len(failures)}")
            for i, failure in enumerate(failures):
                self.logger.warning(f"  Failure {i+1}: {type(failure).__name__}")
        
        if not results:
            self.logger.error("No successful training results to aggregate!")
            return None, {}
        
        self.logger.info(f"Successfully received results from {len(results)} clients")
        
        # Extract parameters and weights
        parameters_list = []
        weights_list = []
        client_metrics = {}
        
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            parameters_list.append(fit_res.parameters)
            weights_list.append(fit_res.num_examples)
            client_metrics[cid] = fit_res.metrics
            
            # Update client quality scores based on training metrics
            if fit_res.metrics and 'train_loss' in fit_res.metrics:
                train_loss = fit_res.metrics['train_loss']
                # Quality score based on loss (lower loss = higher quality)
                quality_score = max(0.1, min(1.0, 1.0 / (1.0 + train_loss)))
                self.client_quality_scores[cid] = quality_score
                
                # Update reliability (successful completion)
                current_reliability = self.client_reliability_scores.get(cid, 0.5)
                self.client_reliability_scores[cid] = 0.9 * current_reliability + 0.1 * 1.0
        
        # Convert parameters to tensors
        def parameters_to_tensors(parameters: Parameters) -> List[torch.Tensor]:
            return [torch.tensor(np.array(param)) for param in parameters.tensors]
        
        # Get current global parameters as tensors
        if self.current_global_params is None:
            # First round - use first client's parameters as baseline
            global_tensors = parameters_to_tensors(parameters_list[0])
        else:
            global_tensors = parameters_to_tensors(self.current_global_params)
        
        # 3.3.2 Pseudo-gradient computation with weighted averaging
        total_examples = sum(weights_list)
        pseudo_gradients = []
        
        for i, param_tensor in enumerate(global_tensors):
            weighted_diff = torch.zeros_like(param_tensor, dtype=torch.float32)
            
            for j, (parameters, weight) in enumerate(zip(parameters_list, weights_list)):
                client_tensors = parameters_to_tensors(parameters)
                if i < len(client_tensors):
                    # Compute pseudo-gradient: global_param - client_param
                    diff = param_tensor.float() - client_tensors[i].float()
                    weighted_diff += (weight / total_examples) * diff
            
            pseudo_gradients.append(weighted_diff)
        
        # 3.3.3 Adam optimization step
        self.t += 1  # Increment time step
        
        # Initialize moment estimates if first round
        if self.m is None:
            self.m = [torch.zeros_like(grad) for grad in pseudo_gradients]
            self.v = [torch.zeros_like(grad) for grad in pseudo_gradients]
        
        # Update moment estimates and apply Adam step
        updated_params = []
        for i, (param, grad) in enumerate(zip(global_tensors, pseudo_gradients)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Apply Adam update with weight decay
            param_float = param.float()
            update = self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
            
            # Add weight decay
            if self.weight_decay > 0:
                update += self.weight_decay * param_float
            
            # Apply update
            updated_param = param_float - update
            
            # Gradient clipping for stability
            param_norm = torch.norm(updated_param)
            if param_norm > 10.0:  # Clip large parameters
                updated_param = updated_param * (10.0 / param_norm)
            
            updated_params.append(updated_param)
        
        # Convert back to Parameters
        updated_parameters = Parameters(
            tensors=[param.detach().numpy() for param in updated_params],
            tensor_type="numpy.ndarray"
        )
        
        # Store current global parameters
        self.current_global_params = updated_parameters
        
        # 3.3.4 Metrics aggregation
        aggregated_metrics = self._aggregate_training_metrics(client_metrics, weights_list)
        
        # Track loss history for adaptive scheduling
        if 'train_loss' in aggregated_metrics:
            self.loss_history.append(aggregated_metrics['train_loss'])
            # Keep only recent history
            if len(self.loss_history) > 10:
                self.loss_history = self.loss_history[-10:]
        
        # Log aggregated metrics
        self.logger.info(f"Aggregated training metrics:")
        for key, value in aggregated_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.4f}")
        
        return updated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results with comprehensive medical metrics
        """
        self.logger.info(f"AGGREGATE EVALUATION RESULTS - Round {server_round}")
        
        # Handle failures
        if failures:
            self.logger.warning(f"Evaluation failures: {len(failures)}")
            for i, failure in enumerate(failures):
                self.logger.warning(f"  Failure {i+1}: {type(failure).__name__}")
        
        if not results:
            self.logger.error(f"No successful evaluation results in round {server_round}")
            return None, {}
        
        # Filter valid results
        valid_results = [res for res in results if res[1].num_examples > 0]
        if not valid_results:
            self.logger.error(f"All evaluation results had 0 examples in round {server_round}")
            return None, {}
        
        self.logger.info(f"Processing evaluation results from {len(valid_results)} clients")
        
        # Extract metrics and compute weighted averages
        total_examples = sum(res[1].num_examples for res in valid_results)
        weighted_loss = 0.0
        aggregated_metrics = {}
        
        # Collect all metric keys
        all_metric_keys = set()
        for _, eval_res in valid_results:
            if eval_res.metrics:
                all_metric_keys.update(eval_res.metrics.keys())
        
        # Aggregate each metric
        for metric_key in all_metric_keys:
            weighted_sum = 0.0
            total_weight = 0
            
            for client_proxy, eval_res in valid_results:
                if eval_res.metrics and metric_key in eval_res.metrics:
                    try:
                        value = float(eval_res.metrics[metric_key])
                        weight = eval_res.num_examples
                        weighted_sum += value * weight
                        total_weight += weight
                    except (ValueError, TypeError):
                        continue
            
            if total_weight > 0:
                aggregated_metrics[metric_key] = weighted_sum / total_weight
        
        # Aggregate loss
        for client_proxy, eval_res in valid_results:
            weight = eval_res.num_examples / total_examples
            weighted_loss += eval_res.loss * weight
        
        # Performance tracking and analysis
        self._track_global_performance(server_round, aggregated_metrics)
        
        # Log comprehensive results
        self.logger.info(f"Round {server_round} Evaluation Summary:")
        self.logger.info(f"  Total examples: {total_examples}")
        self.logger.info(f"  Aggregated loss: {weighted_loss:.4f}")
        
        # Log key medical metrics
        medical_metrics = ['dice_avg', 'dice_foreground_avg', 'iou_avg']
        for metric in medical_metrics:
            if metric in aggregated_metrics:
                self.logger.info(f"  {metric}: {aggregated_metrics[metric]:.4f}")
        
        return weighted_loss, aggregated_metrics
    
    def _aggregate_training_metrics(
        self, 
        client_metrics: Dict[str, Dict], 
        weights: List[int]
    ) -> Dict[str, Scalar]:
        """Aggregate training metrics from multiple clients"""
        if not client_metrics:
            return {}
        
        # Collect all metric keys
        all_keys = set()
        for metrics in client_metrics.values():
            if metrics:
                all_keys.update(metrics.keys())
        
        aggregated = {}
        total_weight = sum(weights)
        
        for key in all_keys:
            weighted_sum = 0.0
            valid_weight = 0
            
            for i, (cid, metrics) in enumerate(client_metrics.items()):
                if metrics and key in metrics:
                    try:
                        value = float(metrics[key])
                        weight = weights[i] if i < len(weights) else 1
                        weighted_sum += value * weight
                        valid_weight += weight
                    except (ValueError, TypeError):
                        continue
            
            if valid_weight > 0:
                aggregated[key] = weighted_sum / valid_weight
        
        return aggregated
    
    def _track_global_performance(self, server_round: int, metrics: Dict[str, Scalar]):
        """Track global model performance and convergence"""
        # Store round metrics
        self.round_metrics[f"round_{server_round}"] = metrics
        
        # Track best performance
        current_metric = 0.0
        if 'dice_foreground_avg' in metrics:
            current_metric = float(metrics['dice_foreground_avg'])
        elif 'dice_avg' in metrics:
            current_metric = float(metrics['dice_avg'])
        
        if current_metric > self.best_global_metric:
            improvement = current_metric - self.best_global_metric
            self.best_global_metric = current_metric
            self.rounds_without_improvement = 0
            self.logger.info(f"🎉 NEW BEST GLOBAL MODEL! Improved by {improvement:.4f}")
        else:
            self.rounds_without_improvement += 1
            self.logger.info(f"No improvement for {self.rounds_without_improvement} rounds")
            
            if self.rounds_without_improvement >= self.max_patience:
                self.logger.warning(f"Early stopping recommended - no improvement for {self.max_patience} rounds")
        
        # Performance trend analysis
        if server_round >= 3:
            self._analyze_performance_trends(server_round)
    
    def _analyze_performance_trends(self, server_round: int):
        """Analyze performance trends over recent rounds"""
        recent_metrics = []
        for r in range(max(1, server_round - 2), server_round + 1):
            round_metrics = self.round_metrics.get(f"round_{r}", {})
            if 'dice_foreground_avg' in round_metrics:
                recent_metrics.append(float(round_metrics['dice_foreground_avg']))
            elif 'dice_avg' in round_metrics:
                recent_metrics.append(float(round_metrics['dice_avg']))
        
        if len(recent_metrics) >= 3:
            trend = recent_metrics[-1] - recent_metrics[0]
            if trend > 0.05:
                self.logger.info(f"📈 Positive trend: +{trend:.3f} over last 3 rounds")
            elif trend < -0.02:
                self.logger.warning(f"📉 Negative trend: {trend:.3f} - consider reducing learning rate")
            else:
                self.logger.info(f"📊 Stable performance: {trend:+.3f} change")

# =============================================================================
# 4. GLOBAL MODEL MANAGEMENT
# =============================================================================

def create_global_model(config: Dict[str, Any]) -> OptimizedRobustMedVFL_UNet:
    """
    Create and initialize global model with optimal configuration
    """
    # Extract model configuration
    num_classes = config.get('num_classes', ServerConfig.NUM_CLASSES)
    dropout_rate = config.get('dropout_rate', 0.1)
    
    # Default balanced pruning configuration
    pruning_config = {
        'noise_processing_levels': [0, 1],
        'maxwell_solver_levels': [0, 1],
        'dropout_positions': [0],
        'skip_quantum_noise': False
    }
    
    # Create model
    model = OptimizedRobustMedVFL_UNet(
        n_channels=1,
        n_classes=num_classes,
        dropout_rate=dropout_rate,
        pruning_config=pruning_config
    ).to(DEVICE)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    print(f"✓ Global model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model

def get_evaluate_fn(
    global_model: OptimizedRobustMedVFL_UNet,
    test_dataloader: Optional[DataLoader],
    config: Dict[str, Any]
) -> Callable:
    """
    Create evaluation function for global model assessment
    """
    def evaluate(server_round: int, parameters: Parameters, config_dict: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on test dataset"""
        if test_dataloader is None:
            print(f"Warning: No test data available for global evaluation")
            return None
        
        # Set model parameters
        params_dict = zip(global_model.state_dict().keys(), parameters.tensors)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        global_model.load_state_dict(state_dict, strict=True)
        
        # Evaluate model
        try:
            metrics = evaluate_metrics(global_model, test_dataloader, DEVICE, ServerConfig.NUM_CLASSES)
            
            # Calculate primary metrics
            avg_dice = np.mean(metrics['dice_scores'])
            fg_dice = np.mean(metrics['dice_scores'][1:]) if len(metrics['dice_scores']) > 1 else avg_dice
            avg_iou = np.mean(metrics['iou'])
            
            # Create evaluation metrics dictionary
            eval_metrics = {
                'global_dice_avg': avg_dice,
                'global_dice_foreground': fg_dice,
                'global_iou_avg': avg_iou,
                'global_precision_avg': np.mean(metrics['precision']),
                'global_recall_avg': np.mean(metrics['recall']),
                'global_f1_avg': np.mean(metrics['f1_score']),
            }
            
            # Add per-class metrics
            for i, class_name in enumerate(ServerConfig.CLASS_NAMES):
                if i < len(metrics['dice_scores']):
                    eval_metrics[f'global_dice_{class_name.lower()}'] = metrics['dice_scores'][i]
                    eval_metrics[f'global_iou_{class_name.lower()}'] = metrics['iou'][i]
            
            # Use negative dice as loss (lower is better)
            loss = 1.0 - fg_dice
            
            print(f"Global evaluation - Round {server_round}:")
            print(f"  Global Dice (FG): {fg_dice:.4f}")
            print(f"  Global IoU: {avg_iou:.4f}")
            
            return loss, eval_metrics
            
        except Exception as e:
            print(f"Error in global evaluation: {e}")
            return None
    
    return evaluate

def load_global_test_data(config: Dict[str, Any]) -> Optional[DataLoader]:
    """
    Load global test dataset for server-side evaluation
    """
    try:
        # Try to load test data from configured path
        test_data_path = config.get('test_data_path', '/Users/alvinluong/Documents/Federated_Learning/ACDC/database/testing')
        
        if os.path.exists(test_data_path):
            print(f"Loading global test data from: {test_data_path}")
            
            # Create test dataloader using ACDC dataset
            test_dataloader = create_acdc_dataloader(
                data_dir=test_data_path,
                batch_size=config.get('test_batch_size', 4),
                shuffle=False,
                augment=False,  # No augmentation for testing
                num_workers=0
            )
            
            print(f"✓ Global test data loaded: {len(test_dataloader.dataset)} samples")
            return test_dataloader
            
        else:
            print(f"Warning: Test data path not found: {test_data_path}")
            return None
            
    except Exception as e:
        print(f"Error loading global test data: {e}")
        return None

# =============================================================================
# 5. CONFIGURATION MANAGEMENT
# =============================================================================

def get_fit_config(
    server_round: int, 
    convergence_status: str, 
    client_capabilities: Dict[str, Any]
) -> Dict[str, Scalar]:
    """
    Get adaptive fit configuration based on round and client capabilities
    """
    # Base configuration
    config = {
        "server_round": server_round,
        "learning_rate": ServerConfig.DEFAULT_LEARNING_RATE,
        "local_epochs": ServerConfig.DEFAULT_LOCAL_EPOCHS,
        "batch_size": ServerConfig.DEFAULT_BATCH_SIZE,
        "num_classes": ServerConfig.NUM_CLASSES,
    }
    
    # Adaptive adjustments based on convergence
    if convergence_status == "poor":
        config["learning_rate"] *= 0.5
        config["local_epochs"] = min(config["local_epochs"] + 1, 8)
    elif convergence_status == "excellent":
        config["local_epochs"] = max(config["local_epochs"] - 1, 3)
    
    # Client capability adjustments
    if client_capabilities.get("low_memory", False):
        config["batch_size"] = max(config["batch_size"] // 2, 2)
    
    if client_capabilities.get("high_compute", False):
        config["local_epochs"] = min(config["local_epochs"] + 1, 10)
    
    return config

def get_evaluate_config(server_round: int, evaluation_depth: str) -> Dict[str, Scalar]:
    """
    Get evaluation configuration based on round and desired depth
    """
    config = {
        "server_round": server_round,
        "num_classes": ServerConfig.NUM_CLASSES,
        "collect_detailed_metrics": evaluation_depth == "comprehensive",
        "collect_per_class_metrics": True,
        "collect_medical_metrics": True,
    }
    
    # Comprehensive evaluation every 5 rounds
    if server_round % 5 == 0:
        config["evaluation_depth"] = "comprehensive"
        config["collect_physics_metrics"] = True
        config["save_predictions"] = True
    else:
        config["evaluation_depth"] = "standard"
        config["collect_physics_metrics"] = False
        config["save_predictions"] = False
    
    return config

# =============================================================================
# 6. SERVER FACTORY FUNCTION
# =============================================================================

def server_fn(context: Context) -> ServerAppComponents:
    """
    Create server components with comprehensive configuration
    Factory function for Flower ServerApp
    """
    print("🚀 INITIALIZING ENHANCED MEDICAL FEDERATED LEARNING SERVER")
    print("="*70)
    
    # 6.1 Context Processing and Validation
    try:
        # Extract configuration from context
        run_config = context.run_config
        
        # Server configuration with defaults
        server_config = {
            'num_rounds': run_config.get('num_rounds', ServerConfig.DEFAULT_ROUNDS),
            'learning_rate': run_config.get('learning_rate', ServerConfig.DEFAULT_LEARNING_RATE),
            'min_fit_clients': run_config.get('min_fit_clients', ServerConfig.MIN_FIT_CLIENTS),
            'min_evaluate_clients': run_config.get('min_evaluate_clients', ServerConfig.MIN_EVALUATE_CLIENTS),
            'fraction_fit': run_config.get('fraction_fit', ServerConfig.DEFAULT_FRACTION_FIT),
            'fraction_evaluate': run_config.get('fraction_evaluate', ServerConfig.DEFAULT_FRACTION_EVALUATE),
            'test_data_path': run_config.get('test_data_path', None),
            'enable_global_evaluation': run_config.get('enable_global_evaluation', True),
            'noise_adaptation': run_config.get('noise_adaptation', True),
            'physics_scheduling': run_config.get('physics_scheduling', True),
            'progressive_complexity': run_config.get('progressive_complexity', True),
        }
        
        print("✓ Configuration extracted from context")
        print(f"  Rounds: {server_config['num_rounds']}")
        print(f"  Learning rate: {server_config['learning_rate']}")
        print(f"  Min fit clients: {server_config['min_fit_clients']}")
        
    except Exception as e:
        print(f"⚠ Error extracting configuration: {e}")
        print("Using default configuration")
        server_config = {
            'num_rounds': ServerConfig.DEFAULT_ROUNDS,
            'learning_rate': ServerConfig.DEFAULT_LEARNING_RATE,
            'min_fit_clients': ServerConfig.MIN_FIT_CLIENTS,
            'min_evaluate_clients': ServerConfig.MIN_EVALUATE_CLIENTS,
            'fraction_fit': ServerConfig.DEFAULT_FRACTION_FIT,
            'fraction_evaluate': ServerConfig.DEFAULT_FRACTION_EVALUATE,
            'enable_global_evaluation': True,
            'noise_adaptation': True,
            'physics_scheduling': True,
            'progressive_complexity': True,
        }
    
    # 6.2 Strategy Initialization
    print("\n🧠 INITIALIZING MEDICAL FEDADAM STRATEGY")
    
    # Global model setup
    global_model = create_global_model(server_config)
    
    # Global test data loading
    test_dataloader = None
    if server_config.get('enable_global_evaluation', True):
        test_dataloader = load_global_test_data(server_config)
    
    # Evaluation function setup
    evaluate_fn = None
    if test_dataloader is not None:
        evaluate_fn = get_evaluate_fn(global_model, test_dataloader, server_config)
        print("✓ Global evaluation function configured")
    else:
        print("⚠ Global evaluation disabled - no test data available")
    
    # 6.3 Server Components Assembly
    # Strategy instantiation
    strategy = MedicalFedAdam(
        learning_rate=server_config['learning_rate'],
        min_fit_clients=server_config['min_fit_clients'],
        min_evaluate_clients=server_config['min_evaluate_clients'],
        fraction_fit=server_config['fraction_fit'],
        fraction_evaluate=server_config['fraction_evaluate'],
        noise_adaptation=server_config['noise_adaptation'],
        physics_scheduling=server_config['physics_scheduling'],
        progressive_complexity=server_config['progressive_complexity'],
        evaluate_fn=evaluate_fn,
        num_rounds=server_config['num_rounds'],
    )
    
    print("✓ MedicalFedAdam strategy initialized")
    
    # Server configuration
    config = ServerConfig(
        num_rounds=server_config['num_rounds'],
        round_timeout=ServerConfig.COMPUTATION_TIMEOUT
    )
    
    print("✓ Server configuration created")
    
    # Components integration
    components = ServerAppComponents(
        strategy=strategy,
        config=config
    )
    
    print("✓ Server components assembled")
    print("="*70)
    print("🌸 MEDICAL FEDERATED LEARNING SERVER READY")
    print(f"📊 Configuration Summary:")
    print(f"  • Rounds: {server_config['num_rounds']}")
    print(f"  • Strategy: MedicalFedAdam with adaptive complexity")
    print(f"  • Global evaluation: {'Enabled' if evaluate_fn else 'Disabled'}")
    print(f"  • Noise adaptation: {'Enabled' if server_config['noise_adaptation'] else 'Disabled'}")
    print(f"  • Physics scheduling: {'Enabled' if server_config['physics_scheduling'] else 'Disabled'}")
    print(f"  • Progressive complexity: {'Enabled' if server_config['progressive_complexity'] else 'Disabled'}")
    print("="*70)
    
    return components

# =============================================================================
# 7. UTILITY FUNCTIONS
# =============================================================================

def analyze_convergence(loss_history: List[float], patience: int = 3) -> Dict[str, Any]:
    """
    Analyze convergence patterns and provide recommendations
    """
    if len(loss_history) < 3:
        return {"status": "insufficient_data", "recommendation": "continue"}
    
    recent_losses = loss_history[-patience:]
    
    # Check for improvement
    if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
        return {"status": "improving", "recommendation": "continue"}
    
    # Check for stagnation
    if max(recent_losses) - min(recent_losses) < 0.001:
        return {"status": "stagnant", "recommendation": "reduce_lr"}
    
    # Check for divergence
    if recent_losses[-1] > recent_losses[0] * 1.1:
        return {"status": "diverging", "recommendation": "reduce_lr_significantly"}
    
    return {"status": "stable", "recommendation": "continue"}

def estimate_remaining_time(round_times: List[float], current_round: int, total_rounds: int) -> str:
    """
    Estimate remaining training time based on historical round times
    """
    if not round_times or current_round >= total_rounds:
        return "Unknown"
    
    avg_time = np.mean(round_times[-5:])  # Use last 5 rounds
    remaining_rounds = total_rounds - current_round
    remaining_seconds = avg_time * remaining_rounds
    
    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

# =============================================================================
# 8. SERVERAPP CREATION AND EXPORT
# =============================================================================

# Create the ServerApp with our factory function
app = ServerApp(server_fn=server_fn)

# =============================================================================
# 9. MAIN EXECUTION AND CLI SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("🌸 ENHANCED MEDICAL FEDERATED LEARNING SERVER v3.0")
    print("="*70)
    print("🔧 ADVANCED FEATURES:")
    print("✓ MedicalFedAdam strategy with adaptive optimization")
    print("✓ Progressive model complexity scheduling")
    print("✓ Medical domain-specific noise adaptation")
    print("✓ Physics-informed constraint scheduling")
    print("✓ Intelligent client selection based on quality")
    print("✓ Comprehensive medical metrics aggregation")
    print("✓ Global model evaluation with clinical metrics")
    print("✓ Convergence analysis and early stopping")
    print("✓ Resource-aware client management")
    print("✓ Production-ready error handling and logging")
    print("")
    print("📊 MEDICAL SPECIALIZATIONS:")
    print("✓ ACDC cardiac segmentation optimization")
    print("✓ Medical-safe data augmentation")
    print("✓ Clinical performance metrics (Dice, IoU)")
    print("✓ Physics-informed Maxwell equation constraints")
    print("✓ Quantum noise injection for robustness")
    print("✓ Adaptive pruning for computational efficiency")
    print("="*70)
    
    print("\n🚀 DEPLOYMENT OPTIONS:")
    print("1. CLI (recommended): flower-server --app fl_core.app_server:app")
    print("2. Direct execution: python app_server.py --start-server")
    print("3. Custom config: flower-server --app fl_core.app_server:app --run-config config.toml")
    
    # Direct execution option for development
    import sys
    if "--start-server" in sys.argv:
        try:
            print("\n🌸 Starting Flower server directly...")
            
            # Create a simple context for direct execution
            from flwr.common import Context
            context = Context(
                run_id=0,
                node_id=0,
                node_config={},
                state={},
                run_config={
                    'num_rounds': 10,
                    'learning_rate': 1e-4,
                    'min_fit_clients': 1,
                    'enable_global_evaluation': True,
                }
            )
            
            # Get server components
            components = server_fn(context)
            
            # Start server
            fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=components.config,
                strategy=components.strategy,
                grpc_max_message_length=ServerConfig.MAX_MESSAGE_LENGTH,
                certificates=None  # No SSL for development
            )
            
        except Exception as e:
            print(f"❌ Server execution failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("\n💡 Use --start-server flag to run directly, or use the CLI command above.")
        print("📖 For production deployment, use the CLI with proper configuration files.")

