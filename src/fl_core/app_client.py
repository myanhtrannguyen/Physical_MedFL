# 1. HEADER AND IMPORTS SECTION

# 1.1 Standard Library Imports
import os
import sys
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import json
import pickle
import time
import gc
from pathlib import Path

# 1.2 Scientific Computing Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

# 1.3 Flower Framework Imports
from flwr.client import ClientApp, NumPyClient, Client
from flwr.common import Context, Config, NDArrays, Scalar

# 1.4 Project-Specific Imports
try:
    from src.models.mlp_model import OptimizedRobustMedVFL_UNet
    from src.models.model_factory import create_model, get_model_info
    from src.data.dataset import ACDCDataset, MedicalSegmentationDataset
    from src.data.loader import create_acdc_dataloader, create_dataloader_from_paths
    from src.data.preprocessing import MedicalImagePreprocessor, DataAugmentation
    from src.utils.seed import set_seed
    from src.utils.logger import setup_federated_logger
    SRC_IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Failed to import from src modules: {e}")
    SRC_IMPORTS_AVAILABLE = False
    # Fallback imports for backward compatibility
    try:
        from models.mlp_model import RobustMedVFL_UNet  # type: ignore
        # Define fallback functions
        def get_model_info(model):
            """Fallback model info function"""
            total_params = sum(p.numel() for p in model.parameters())
            return {"total_parameters": total_params, "model_size_mb": total_params * 4 / (1024 * 1024)}
        
        def setup_federated_logger(client_id=None, log_dir="logs", level=logging.INFO):
            """Fallback logger setup"""
            logger = logging.getLogger(f"client_{client_id}" if client_id else "client")
            logger.setLevel(level)
            return logger
        
        def set_seed(seed):
            """Fallback seed function"""
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Define fallback data classes
        class FallbackMedicalImagePreprocessor:
            def __init__(self, **kwargs):
                pass
        
        class FallbackDataAugmentation:
            def __init__(self, **kwargs):
                pass
        
        # Assign fallback classes to expected names
        MedicalImagePreprocessor = FallbackMedicalImagePreprocessor
        DataAugmentation = FallbackDataAugmentation
        
        def create_acdc_dataloader(*args, **kwargs) -> DataLoader:
            """Fallback dataloader creation"""
            raise NotImplementedError("ACDC dataloader not available without src imports")
        
        def create_dataloader_from_paths(*args, **kwargs) -> DataLoader:
            """Fallback dataloader creation"""
            raise NotImplementedError("Dataloader from paths not available without src imports")
            
    except ImportError:
        logging.error("Could not import RobustMedVFL_UNet from any location")
        raise

# 1.5 Model Components Imports (with fallback handling)
try:
    # Try to import advanced components from MLP_Model
    from mlp_model import (  # type: ignore
        CombinedLoss, 
        quantum_noise_injection,
        MaxwellSolver, 
        ePURE, 
        adaptive_spline_smoothing
    )
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced components not available: {e}")
    ADVANCED_COMPONENTS_AVAILABLE = False
    # Define fallback implementations
    def quantum_noise_injection(x, factor=0.01):
        """Fallback quantum noise injection"""
        return x + torch.randn_like(x) * factor
    
    class MaxwellSolver:
        """Fallback Maxwell solver"""
        def __init__(self, *args, **kwargs):
            pass
        def solve(self, x):
            return torch.zeros_like(x)
    
    class ePURE:
        """Fallback ePURE implementation"""
        def __init__(self, *args, **kwargs):
            pass
        def estimate_noise(self, x):
            return torch.zeros_like(x)
    
    class CombinedLoss:
        """Fallback combined loss"""
        def __init__(self, *args, **kwargs):
            self.criterion = nn.CrossEntropyLoss()
        def __call__(self, outputs, targets):
            return {"total_loss": self.criterion(outputs, targets)}
    
    def adaptive_spline_smoothing(x):
        """Fallback spline smoothing"""
        return x

# 2. GLOBAL CONFIGURATION AND SETUP

# 2.1 Environment Setup
# Suppress gRPC warnings for clean logging
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '2')
warnings.filterwarnings('ignore', category=UserWarning)

# Device detection with CUDA availability check
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {DEVICE}")

# 2.2 Constants Definition
# Model hyperparameters
DEFAULT_MODEL_CONFIG = {
    'n_channels': 1,
    'n_classes': 4,
    'dropout_rate': 0.1,
    'use_batch_norm': True,
    'use_residual': True
}

# Training constants
DEFAULT_TRAINING_CONFIG = {
    'local_epochs': 5,
    'batch_size': 8,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'gradient_clip_norm': 1.0
}

# Medical imaging constants
MEDICAL_CONFIG = {
    'img_height': 256,
    'img_width': 256,
    'num_classes': 4,
    'target_spacing': (1.0, 1.0),
    'intensity_range': (0, 1)
}

# Noise parameters
NOISE_CONFIG = {
    'quantum_noise_factor': 0.01,
    'dropout_rate': 0.1,
    'noise_schedule': 'adaptive',
    'min_noise': 0.001,
    'max_noise': 0.05
}

# 3. FLOWERCLIENT CLASS DEFINITION

class FlowerClient(NumPyClient):
    """
    Comprehensive Federated Learning Client with Advanced Medical Imaging Components
    
    Features:
    - Quantum noise injection for robustness
    - Maxwell solver for physics constraints
    - ePURE noise estimation
    - Medical data augmentation
    - Comprehensive metrics tracking
    - Resource monitoring
    """
    
    def __init__(
        self,
        client_id: str,
        data_path: str,
        model_config: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize comprehensive federated client.
        
        Args:
            client_id: Unique client identifier
            data_path: Path to client's data partition
            model_config: Model configuration parameters
            training_config: Training configuration parameters
            device: Computing device (CPU/GPU)
        """
        # 3.1.1 Parameter Initialization
        self.client_id = client_id
        self.data_path = Path(data_path)
        self.device = device or DEVICE
        
        # Configuration setup
        self.model_config = {**DEFAULT_MODEL_CONFIG, **(model_config or {})}
        self.training_config = {**DEFAULT_TRAINING_CONFIG, **(training_config or {})}
        
        # Setup logging
        self.logger = setup_federated_logger(
            client_id=client_id,
            log_dir="logs/clients",
            level=logging.INFO
        )
        
        # 3.1.2 Model Component Configuration
        self._initialize_model()
        self._initialize_advanced_components()
        self._initialize_data_components()
        
        # 3.1.3 Training State Management
        self.metrics_history = {
            'loss': [],
            'accuracy': [],
            'dice_score': [],
            'iou_score': [],
            'physics_loss': [],
            'noise_level': []
        }
        
        self.current_round = 0
        self.total_rounds = 0
        self.convergence_threshold = 1e-4
        self.patience = 5
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Resource monitoring
        self.resource_stats = {
            'memory_usage': [],
            'computation_time': [],
            'communication_time': []
        }
        
        self.logger.info(f"Client {client_id} initialized successfully")
    
    def _initialize_model(self):
        """Initialize the main model with all components."""
        try:
            self.logger.info("Starting model initialization...")
            
            if SRC_IMPORTS_AVAILABLE:
                self.logger.info("Using SRC imports for model")
                from src.models.mlp_model import OptimizedRobustMedVFL_UNet
                
                # Debug model config
                self.logger.info(f"Model config: {self.model_config}")
                
                self.model = OptimizedRobustMedVFL_UNet(
                    n_channels=self.model_config['n_channels'],
                    n_classes=self.model_config['n_classes'],
                    dropout_rate=self.model_config['dropout_rate'],
                    pruning_config={
                        'noise_processing_levels': [0, 1],
                        'maxwell_solver_levels': [0, 1],
                        'dropout_positions': [0],
                        'skip_quantum_noise': False
                    }
                ).to(self.device)
                
                # CRITICAL: Ensure model is in training mode and parameters are trainable
                self.model.train()
                
                # Debug: List all parameters and their requires_grad
                for name, param in self.model.named_parameters():
                    self.logger.info(f"Param: {name}, shape: {tuple(param.shape)}, requires_grad: {param.requires_grad}")
                num_total = sum(1 for _ in self.model.parameters())
                num_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
                self.logger.info(f"[DEBUG] Model has {num_total} parameters, {num_trainable} trainable.")
                
                # Final verification
                if num_trainable == 0:
                    self.logger.error("CRITICAL: All parameters still have requires_grad=False after manual setting!")
                    # Try alternative approach
                    for param in self.model.parameters():
                        param.requires_grad_(True)
                    trainable_final = sum(1 for p in self.model.parameters() if p.requires_grad)
                    self.logger.info(f"After requires_grad_(True): {trainable_final}/{num_total} trainable")
                
                self.logger.info("OptimizedRobustMedVFL_UNet created successfully")
                
            else:
                self.logger.info("Using fallback model imports")
                # Create a simple CNN model as fallback
                class SimpleCNN(nn.Module):
                    def __init__(self, n_channels=1, n_classes=4):
                        super().__init__()
                        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
                        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                        self.conv3 = nn.Conv2d(64, n_classes, kernel_size=1)
                        self.pool = nn.AdaptiveAvgPool2d((64, 64))
                        
                    def forward(self, x):
                        x = F.relu(self.conv1(x))
                        x = F.relu(self.conv2(x))
                        x = self.conv3(x)
                        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
                        return x
                
                self.model = SimpleCNN(
                    n_channels=self.model_config['n_channels'],
                    n_classes=self.model_config['n_classes']
                ).to(self.device)
                
                # CRITICAL: Ensure model is in training mode and parameters are trainable
                self.model.train()
                
                # Debug: List all parameters and their requires_grad
                for name, param in self.model.named_parameters():
                    self.logger.info(f"Param: {name}, shape: {tuple(param.shape)}, requires_grad: {param.requires_grad}")
                num_total = sum(1 for _ in self.model.parameters())
                num_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
                self.logger.info(f"[DEBUG] Model has {num_total} parameters, {num_trainable} trainable.")
                
                # Final verification
                if num_trainable == 0:
                    self.logger.error("CRITICAL: All parameters still have requires_grad=False after manual setting!")
                    # Try alternative approach
                    for param in self.model.parameters():
                        param.requires_grad_(True)
                    trainable_final = sum(1 for p in self.model.parameters() if p.requires_grad)
                    self.logger.info(f"After requires_grad_(True): {trainable_final}/{num_total} trainable")
                
                self.logger.info("SimpleCNN fallback model created")
            
            # Get model information
            self.model_info = get_model_info(self.model)
            self.logger.info(f"Model initialized successfully: {self.model_info}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _initialize_advanced_components(self):
        """Initialize advanced model components."""
        if ADVANCED_COMPONENTS_AVAILABLE:
            # Quantum noise injection setup
            self.quantum_noise_enabled = True
            self.noise_factor = NOISE_CONFIG['quantum_noise_factor']
            
            # Maxwell solver initialization
            try:
                self.maxwell_solver = MaxwellSolver(
                    grid_size=(MEDICAL_CONFIG['img_height'], MEDICAL_CONFIG['img_width']),
                    device=self.device
                )
                self.maxwell_enabled = True
                self.logger.info("Maxwell solver initialized")
            except Exception as e:
                self.logger.warning(f"Maxwell solver initialization failed: {e}")
                self.maxwell_enabled = False
            
            # ePURE noise estimation setup
            try:
                self.epure = ePURE(
                    input_channels=self.model_config['n_channels'],
                    device=self.device
                )
                self.epure_enabled = True
                self.logger.info("ePURE noise estimator initialized")
            except Exception as e:
                self.logger.warning(f"ePURE initialization failed: {e}")
                self.epure_enabled = False
        else:
            self.quantum_noise_enabled = False
            self.maxwell_enabled = False
            self.epure_enabled = False
            self.logger.warning("Advanced components not available, using fallbacks")
    
    def _initialize_data_components(self):
        """Initialize data preprocessing and augmentation components."""
        # Medical image preprocessor
        self.preprocessor = MedicalImagePreprocessor(
            target_size=(MEDICAL_CONFIG['img_height'], MEDICAL_CONFIG['img_width']),
            normalize=True,
            clip_percentiles=(1, 99),
            apply_clahe=True
        )
        
        # Medical data augmentation
        self.augmentation = DataAugmentation(
            rotation_range=15.0,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=0.1,
            contrast_range=0.1,
            noise_std=0.01
        )
        
        # Initialize data loaders as None (will be set when data is loaded)
        self.trainloader = None
        self.valloader = None
        self.num_examples = {"trainset": 0, "valset": 0}
        
        self.logger.info("Data components initialized")
    
    # 3.2 PARAMETER MANAGEMENT METHODS
    
    def get_parameters(self, config: Config) -> NDArrays:
        """
        Extract model parameters for federated aggregation.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of numpy arrays representing model parameters
        """
        try:
            start_time = time.time()
            
            # Extract all model weights from state_dict
            state_dict = self.model.state_dict()
            parameters = []
            
            # Debug: Check what's in state_dict
            self.logger.info(f"State dict has {len(state_dict)} parameters")
            trainable_count = 0
            non_trainable_count = 0
            
            # Convert PyTorch tensors to numpy arrays efficiently
            for key, tensor in state_dict.items():
                if tensor.requires_grad:  # Only include trainable parameters
                    param_array = tensor.detach().cpu().numpy()
                    parameters.append(param_array)
                    trainable_count += 1
                    self.logger.debug(f"Trainable parameter {key}: shape {param_array.shape}")
                else:
                    non_trainable_count += 1
                    self.logger.debug(f"Non-trainable parameter {key}: shape {tensor.shape}")
            
            self.logger.info(f"Found {trainable_count} trainable, {non_trainable_count} non-trainable parameters")
            
            # If no trainable parameters found, fallback to all parameters
            if len(parameters) == 0:
                self.logger.warning("No trainable parameters found! Using all parameters as fallback")
                for key, tensor in state_dict.items():
                    param_array = tensor.detach().cpu().numpy()
                    parameters.append(param_array)
                    self.logger.debug(f"Fallback parameter {key}: shape {param_array.shape}")
            
            # Log parameter extraction info
            extraction_time = time.time() - start_time
            total_params = sum(p.size for p in parameters)
            self.logger.info(f"Extracted {len(parameters)} parameter arrays "
                           f"({total_params:,} total parameters) in {extraction_time:.3f}s")
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error extracting parameters: {e}")
            raise
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model parameters from federated aggregation.
        
        Args:
            parameters: List of numpy arrays representing model parameters
        """
        try:
            start_time = time.time()
            
            # Get current state dict for parameter validation
            state_dict = self.model.state_dict()
            trainable_keys = [key for key, tensor in state_dict.items() if tensor.requires_grad]
            
            # Use same strategy as get_parameters: if no trainable params, use all
            if len(trainable_keys) == 0:
                self.logger.warning("No trainable parameters found in set_parameters! Using all parameters as fallback")
                all_keys = list(state_dict.keys())
            else:
                all_keys = trainable_keys
            
            # Validate parameter count
            if len(parameters) != len(all_keys):
                raise ValueError(f"Parameter count mismatch: expected {len(all_keys)}, "
                               f"got {len(parameters)}")
            
            # Rebuild state_dict from numpy arrays
            new_state_dict = OrderedDict()
            param_idx = 0
            
            for key, tensor in state_dict.items():
                if (len(trainable_keys) > 0 and tensor.requires_grad) or (len(trainable_keys) == 0):
                    # Convert numpy array back to tensor
                    param_tensor = torch.from_numpy(parameters[param_idx]).to(self.device)
                    
                    # Validate shape compatibility
                    if param_tensor.shape != tensor.shape:
                        raise ValueError(f"Shape mismatch for {key}: expected {tensor.shape}, "
                                       f"got {param_tensor.shape}")
                    
                    new_state_dict[key] = param_tensor
                    param_idx += 1
                else:
                    # Keep non-trainable parameters unchanged
                    new_state_dict[key] = tensor
            
            # Load parameters into model with strict checking
            self.model.load_state_dict(new_state_dict, strict=True)
            
            # Verify successful parameter loading
            loading_time = time.time() - start_time
            self.logger.info(f"Successfully loaded {len(parameters)} parameter arrays "
                           f"in {loading_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Error setting parameters: {e}")
            raise
    
    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """
        Report client capabilities and information.
        
        Args:
            config: Configuration from server
            
        Returns:
            Dictionary with client properties
        """
        try:
            # Client capabilities
            properties = {
                # Computational resources
                "device_type": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "memory_total": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                "cpu_count": os.cpu_count(),
                
                # Data information
                "trainset_size": self.num_examples["trainset"],
                "valset_size": self.num_examples["valset"],
                "data_path": str(self.data_path),
                
                # Model configuration
                "model_parameters": self.model_info.get("total_parameters", 0),
                "model_size_mb": self.model_info.get("model_size_mb", 0),
                
                # Advanced components status
                "quantum_noise_enabled": self.quantum_noise_enabled,
                "maxwell_solver_enabled": self.maxwell_enabled,
                "epure_enabled": self.epure_enabled,
                
                # Performance metrics
                "best_loss": self.best_loss,
                "current_round": self.current_round,
                "convergence_patience": self.patience_counter,
                
                # Client version info
                "client_version": "1.0.0",
                "flower_version": "1.8.0"
            }
            
            self.logger.debug(f"Client properties: {properties}")
            return properties
            
        except Exception as e:
            self.logger.error(f"Error getting properties: {e}")
            return {"error": str(e)}
    
    # 3.3 TRAINING METHOD (FIT) - COMPREHENSIVE IMPLEMENTATION
    
    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Comprehensive training implementation with all advanced components.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        import traceback
        try:
            self.logger.info(f"[fit] Start. Parameters: {type(parameters)}, Config: {config}")
            fit_start_time = time.time()
            self.logger.info(f"Starting fit for round {config.get('server_round', 'unknown')}")
            
            # 3.3.1 Pre-training Setup
            self._apply_server_config(config)
            self.set_parameters(parameters)
            
            # Load data if not already loaded
            if self.trainloader is None:
                self._load_client_data()
            
            # 3.3.2 Model Adaptation
            self._adapt_model_for_round(config)
            
            # 3.3.3 Training Loop Implementation
            training_metrics = self._execute_training_loop(config)
            
            # 3.3.4 Training Monitoring and Cleanup
            self._update_training_history(training_metrics)
            self._cleanup_memory()
            
            # 3.3.5 Post-training Processing
            self.logger.info(f"[fit] Training finished. Preparing return values...")
            updated_parameters = self.get_parameters(config)
            return_metrics = {
                "train_loss": float(training_metrics["final_loss"]),
                "train_accuracy": float(training_metrics["final_accuracy"]),
                "train_dice": float(training_metrics["final_dice"]),
                "train_iou": float(training_metrics["final_iou"]),
                "epochs_completed": int(training_metrics["epochs_completed"]),
                "training_time": float(time.time() - fit_start_time),
                "convergence_status": bool(training_metrics["converged"]),
                "noise_level": float(training_metrics["avg_noise_level"]),
                "physics_loss": float(training_metrics["avg_physics_loss"])
            }
            self.logger.info(f"[fit] Return types: {type(updated_parameters)}, {type(self.num_examples['trainset'])}, {type(return_metrics)}")
            self.logger.info(f"[fit] Return values: num_examples={self.num_examples['trainset']}, metrics={return_metrics}")
            return updated_parameters, int(self.num_examples["trainset"]), return_metrics
        except Exception as e:
            self.logger.error(f"[fit] Error: {e}")
            self.logger.error(traceback.format_exc())
            return self.get_parameters(config), 0, {"error": str(e), "train_loss": float('inf')}
    
    def _apply_server_config(self, config: Config):
        """Apply server configuration for this training round."""
        # Extract configuration with defaults and proper type casting
        local_epochs = config.get("local_epochs", self.training_config["local_epochs"])
        self.local_epochs = int(local_epochs) if local_epochs is not None else self.training_config["local_epochs"]
        
        learning_rate = config.get("learning_rate", self.training_config["learning_rate"])
        self.learning_rate = float(learning_rate) if learning_rate is not None else self.training_config["learning_rate"]
        
        batch_size = config.get("batch_size", self.training_config["batch_size"])
        self.batch_size = int(batch_size) if batch_size is not None else self.training_config["batch_size"]
        
        quantum_noise_factor = config.get("quantum_noise_factor", NOISE_CONFIG["quantum_noise_factor"])
        self.quantum_noise_factor = float(quantum_noise_factor) if quantum_noise_factor is not None else NOISE_CONFIG["quantum_noise_factor"]
        
        dropout_rate = config.get("dropout_rate", self.model_config["dropout_rate"])
        self.dropout_rate = float(dropout_rate) if dropout_rate is not None else self.model_config["dropout_rate"]
        
        use_maxwell_solver = config.get("use_maxwell_solver", self.maxwell_enabled)
        self.use_maxwell_solver = bool(use_maxwell_solver) if use_maxwell_solver is not None else self.maxwell_enabled
        
        server_round = config.get("server_round", 0)
        self.server_round = int(server_round) if server_round is not None else 0
        
        self.current_round = self.server_round
        self.logger.info(f"Applied server config for round {self.server_round}")
    
    def _adapt_model_for_round(self, config: Config):
        """Adapt model configuration for current round."""
        # Adaptive noise factor based on round
        if self.quantum_noise_enabled:
            # Decrease noise as training progresses
            round_factor = max(0.1, 1.0 - (self.server_round / 100.0))
            self.noise_factor = self.quantum_noise_factor * round_factor
        
        # Setup optimizer with current learning rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.training_config["weight_decay"]
        )
        
        # Setup loss function
        self.criterion = self._create_combined_loss()
        
        self.logger.info(f"Model adapted for round {self.server_round}")
    
    def _create_combined_loss(self):
        """Create combined loss function with all components."""
        try:
            if ADVANCED_COMPONENTS_AVAILABLE:
                return CombinedLoss(
                    num_classes=self.model_config["n_classes"],
                    device=self.device
                )
            else:
                # Fallback to standard loss
                return nn.CrossEntropyLoss()
        except Exception as e:
            self.logger.warning(f"Failed to create combined loss: {e}, using CrossEntropyLoss")
            return nn.CrossEntropyLoss()
    
    def _execute_training_loop(self, config: Config) -> Dict[str, Any]:
        """Execute the main training loop with all components."""
        try:
            self.model.train()
            
            epoch_losses = []
            epoch_accuracies = []
            epoch_dice_scores = []
            epoch_iou_scores = []
            epoch_physics_losses = []
            epoch_noise_levels = []
            
            converged = False
            
            # Load data if not already loaded
            if self.trainloader is None:
                self.logger.info("Trainloader is None, loading data...")
                self._load_client_data()
                
            if self.trainloader is None:
                self.logger.error("Still no trainloader after loading data!")
                return {
                    "final_loss": 999.0,
                    "final_accuracy": 0.0,
                    "final_dice": 0.0,
                    "final_iou": 0.0,
                    "epochs_completed": 0,
                    "converged": False,
                    "avg_noise_level": 0.0,
                    "avg_physics_loss": 0.0,
                    "loss_history": [999.0],
                    "accuracy_history": [0.0]
                }
            
            self.logger.info(f"Starting training loop with {self.local_epochs} epochs")
            
            for epoch in range(self.local_epochs):
                epoch_start_time = time.time()
                
                batch_losses = []
                batch_accuracies = []
                batch_dice_scores = []
                batch_iou_scores = []
                batch_physics_losses = []
                batch_noise_levels = []
                
                self.logger.info(f"Epoch {epoch+1}/{self.local_epochs} starting...")
                
                for batch_idx, batch_data in enumerate(self.trainloader):
                    try:
                        # Safely unpack batch data
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                            images, masks = batch_data[0], batch_data[1]
                        else:
                            self.logger.error(f"Invalid batch data format: {type(batch_data)}")
                            continue
                            
                        # Move to device
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        
                        # Debug shapes
                        if batch_idx == 0:
                            self.logger.info(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
                        
                        # Apply quantum noise injection
                        if self.quantum_noise_enabled:
                            noise_level = self.noise_factor
                            images = quantum_noise_injection(images, factor=noise_level)
                            batch_noise_levels.append(noise_level)
                        
                        # Zero gradients
                        self.optimizer.zero_grad()
                        
                        # Forward pass with all components
                        outputs = self.model(images)
                        
                        # Handle model output (may be tuple from OptimizedRobustMedVFL_UNet)
                        if isinstance(outputs, tuple):
                            # Take the main output (first element)
                            outputs = outputs[0]
                        
                        # Debug output shape
                        if batch_idx == 0:
                            self.logger.info(f"Model output shape: {outputs.shape}")
                        
                        # ePURE noise estimation
                        if self.epure_enabled:
                            noise_estimate = self.epure.estimate_noise(images)
                            # Apply noise correction (simplified)
                            outputs = outputs - 0.1 * noise_estimate
                        
                        # Maxwell solver physics constraints
                        physics_loss = 0.0
                        if self.maxwell_enabled and self.use_maxwell_solver:
                            physics_constraint = self.maxwell_solver.solve(outputs)
                            physics_loss = torch.mean(physics_constraint ** 2)
                            batch_physics_losses.append(physics_loss.item())
                        
                        # Combined loss computation
                        if isinstance(self.criterion, nn.CrossEntropyLoss):
                            # Standard loss
                            loss = self.criterion(outputs, masks)
                        else:
                            # Advanced combined loss
                            loss_components = self.criterion(outputs, masks)
                            loss = loss_components["total_loss"]
                            if physics_loss > 0:
                                loss = loss + 0.1 * physics_loss
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.training_config["gradient_clip_norm"]
                        )
                        
                        # Optimizer step
                        self.optimizer.step()
                        
                        # Compute metrics
                        with torch.no_grad():
                            # Accuracy
                            predicted = torch.argmax(outputs, dim=1)
                            accuracy = (predicted == masks).float().mean().item()
                            
                            # Dice score (simplified)
                            dice_score = self._compute_dice_score(predicted, masks)
                            
                            # IoU score (simplified)
                            iou_score = self._compute_iou_score(predicted, masks)
                        
                        # Store batch metrics
                        batch_losses.append(loss.item())
                        batch_accuracies.append(accuracy)
                        batch_dice_scores.append(dice_score)
                        batch_iou_scores.append(iou_score)
                        
                        # Log progress
                        if batch_idx % 10 == 0:
                            self.logger.debug(f"Epoch {epoch+1}/{self.local_epochs}, "
                                            f"Batch {batch_idx}/{len(self.trainloader)}, "
                                            f"Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
                                            
                    except Exception as batch_error:
                        self.logger.error(f"Error in batch {batch_idx}: {batch_error}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        continue
                
                # Check if we have any successful batches
                if not batch_losses:
                    self.logger.error(f"No successful batches in epoch {epoch+1}")
                    break
                
                # Epoch metrics
                epoch_loss = np.mean(batch_losses)
                epoch_accuracy = np.mean(batch_accuracies)
                epoch_dice = np.mean(batch_dice_scores)
                epoch_iou = np.mean(batch_iou_scores)
                epoch_physics_loss = np.mean(batch_physics_losses) if batch_physics_losses else 0.0
                epoch_noise_level = np.mean(batch_noise_levels) if batch_noise_levels else 0.0
                
                # Store epoch metrics
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)
                epoch_dice_scores.append(epoch_dice)
                epoch_iou_scores.append(epoch_iou)
                epoch_physics_losses.append(epoch_physics_loss)
                epoch_noise_levels.append(epoch_noise_level)
                
                epoch_time = time.time() - epoch_start_time
                
                self.logger.info(f"Epoch {epoch+1}/{self.local_epochs} completed in {epoch_time:.2f}s - "
                               f"Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, "
                               f"Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}")
                
                # Check for convergence
                if epoch_loss < self.best_loss - self.convergence_threshold:
                    self.best_loss = epoch_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    converged = True
                    break
            
            # Ensure we have at least one epoch result
            if not epoch_losses:
                epoch_losses = [999.0]
                epoch_accuracies = [0.0]
                epoch_dice_scores = [0.0]
                epoch_iou_scores = [0.0]
                epoch_physics_losses = [0.0]
                epoch_noise_levels = [0.0]
            
            # Return training metrics
            return {
                "final_loss": float(epoch_losses[-1]),
                "final_accuracy": float(epoch_accuracies[-1]),
                "final_dice": float(epoch_dice_scores[-1]),
                "final_iou": float(epoch_iou_scores[-1]),
                "epochs_completed": len(epoch_losses),
                "converged": converged,
                "avg_noise_level": float(np.mean(epoch_noise_levels)) if epoch_noise_levels else 0.0,
                "avg_physics_loss": float(np.mean(epoch_physics_losses)) if epoch_physics_losses else 0.0,
                "loss_history": [float(x) for x in epoch_losses],
                "accuracy_history": [float(x) for x in epoch_accuracies]
            }
            
        except Exception as training_error:
            self.logger.error(f"Critical error in training loop: {training_error}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return safe fallback values
            return {
                "final_loss": 999.0,
                "final_accuracy": 0.0,
                "final_dice": 0.0,
                "final_iou": 0.0,
                "epochs_completed": 0,
                "converged": False,
                "avg_noise_level": 0.0,
                "avg_physics_loss": 0.0,
                "loss_history": [999.0],
                "accuracy_history": [0.0]
            }
    
    def _compute_dice_score(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Dice score for segmentation."""
        try:
            # Convert to binary for each class and compute Dice
            dice_scores = []
            for class_id in range(self.model_config["n_classes"]):
                pred_class = (predicted == class_id).float()
                target_class = (target == class_id).float()
                
                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum()
                
                if union > 0:
                    dice = (2.0 * intersection) / union
                    dice_scores.append(dice.item())
            
            return float(np.mean(dice_scores)) if dice_scores else 0.0
        except Exception:
            return 0.0
    
    def _compute_iou_score(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Compute IoU score for segmentation."""
        try:
            # Convert to binary for each class and compute IoU
            iou_scores = []
            for class_id in range(self.model_config["n_classes"]):
                pred_class = (predicted == class_id).float()
                target_class = (target == class_id).float()
                
                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum() - intersection
                
                if union > 0:
                    iou = intersection / union
                    iou_scores.append(iou.item())
            
            return float(np.mean(iou_scores)) if iou_scores else 0.0
        except Exception:
            return 0.0
    
    def _update_training_history(self, training_metrics: Dict[str, Any]):
        """Update training history with current metrics."""
        self.metrics_history["loss"].extend(training_metrics["loss_history"])
        self.metrics_history["accuracy"].extend(training_metrics["accuracy_history"])
        self.metrics_history["dice_score"].append(training_metrics["final_dice"])
        self.metrics_history["iou_score"].append(training_metrics["final_iou"])
        self.metrics_history["physics_loss"].append(training_metrics["avg_physics_loss"])
        self.metrics_history["noise_level"].append(training_metrics["avg_noise_level"])
    
    def _cleanup_memory(self):
        """Clean up memory after training."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # 3.4 EVALUATION METHOD (EVALUATE) - DETAILED IMPLEMENTATION
    
    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Comprehensive evaluation implementation.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        import traceback
        try:
            self.logger.info(f"[evaluate] Start. Parameters: {type(parameters)}, Config: {config}")
            eval_start_time = time.time()
            self.logger.info("Starting evaluation")
            
            # 3.4.1 Pre-evaluation Setup
            self.set_parameters(parameters)
            self.model.eval()
            
            # Load validation data if not available
            if self.valloader is None:
                self._load_client_data()
            
            if self.valloader is None:
                self.logger.warning("No validation data available")
                return float('inf'), 0, {"error": "No validation data"}
            
            # 3.4.2 Evaluation Loop
            eval_metrics = self._execute_evaluation_loop()
            
            # 3.4.3 Comprehensive Metrics Collection
            evaluation_time = time.time() - eval_start_time
            
            # 3.4.4 Results Processing
            self.logger.info(f"[evaluate] Evaluation finished. Preparing return values...")
            return_metrics = {
                "eval_loss": float(eval_metrics["avg_loss"]),
                "eval_accuracy": float(eval_metrics["avg_accuracy"]),
                "eval_dice": float(eval_metrics["avg_dice"]),
                "eval_iou": float(eval_metrics["avg_iou"]),
                "eval_precision": float(eval_metrics["avg_precision"]),
                "eval_recall": float(eval_metrics["avg_recall"]),
                "eval_f1": float(eval_metrics["avg_f1"]),
                "physics_consistency": float(eval_metrics["physics_consistency"]),
                "noise_robustness": float(eval_metrics["noise_robustness"]),
                "evaluation_time": float(time.time() - eval_start_time),
                "inference_time_per_sample": float(eval_metrics["avg_inference_time"]),
                "memory_usage_mb": float(eval_metrics["peak_memory_mb"])
            }
            self.logger.info(f"[evaluate] Return types: {type(eval_metrics['avg_loss'])}, {type(self.num_examples['valset'])}, {type(return_metrics)}")
            self.logger.info(f"[evaluate] Return values: num_examples={self.num_examples['valset']}, metrics={return_metrics}")
            return float(eval_metrics["avg_loss"]), int(self.num_examples["valset"]), dict(return_metrics)
        except Exception as e:
            self.logger.error(f"[evaluate] Error: {e}")
            self.logger.error(traceback.format_exc())
            return float('inf'), 0, {"error": str(e)}
    
    def _execute_evaluation_loop(self) -> Dict[str, Any]:
        """Execute comprehensive evaluation loop."""
        all_losses = []
        all_accuracies = []
        all_dice_scores = []
        all_iou_scores = []
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        physics_consistencies = []
        noise_robustness_scores = []
        inference_times = []
        
        peak_memory = 0
        
        with torch.no_grad():
            if self.valloader is None:
                return {
                    "avg_loss": float('inf'),
                    "avg_accuracy": 0.0,
                    "avg_dice": 0.0,
                    "avg_iou": 0.0,
                    "avg_precision": 0.0,
                    "avg_recall": 0.0,
                    "avg_f1": 0.0,
                    "physics_consistency": 1.0,
                    "noise_robustness": 1.0,
                    "avg_inference_time": 0.0,
                    "peak_memory_mb": 0.0
                }
            
            for batch_idx, (images, masks) in enumerate(self.valloader):
                batch_start_time = time.time()
                
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass without noise injection for clean evaluation
                outputs = self.model(images)
                
                # Handle model output (may be tuple from OptimizedRobustMedVFL_UNet)
                if isinstance(outputs, tuple):
                    # Take the main output (first element)
                    outputs = outputs[0]
                
                # Compute loss
                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    loss = self.criterion(outputs, masks)
                else:
                    loss_components = self.criterion(outputs, masks)
                    loss = loss_components["total_loss"]
                
                # Compute predictions
                predicted = torch.argmax(outputs, dim=1)
                
                # Compute metrics
                accuracy = (predicted == masks).float().mean().item()
                dice_score = self._compute_dice_score(predicted, masks)
                iou_score = self._compute_iou_score(predicted, masks)
                
                # Compute precision, recall, F1 per class
                precision, recall, f1 = self._compute_classification_metrics(predicted, masks)
                
                # Physics consistency check
                if self.maxwell_enabled:
                    physics_consistency = self._evaluate_physics_consistency(outputs)
                    physics_consistencies.append(physics_consistency)
                
                # Noise robustness test
                noise_robustness = self._evaluate_noise_robustness(images, masks)
                noise_robustness_scores.append(noise_robustness)
                
                # Store metrics
                all_losses.append(loss.item())
                all_accuracies.append(accuracy)
                all_dice_scores.append(dice_score)
                all_iou_scores.append(iou_score)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1_scores.append(f1)
                
                # Track inference time
                inference_time = time.time() - batch_start_time
                inference_times.append(inference_time)
                
                # Track memory usage
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    peak_memory = max(peak_memory, current_memory)
        
        # Aggregate metrics
        return {
            "avg_loss": np.mean(all_losses),
            "avg_accuracy": np.mean(all_accuracies),
            "avg_dice": np.mean(all_dice_scores),
            "avg_iou": np.mean(all_iou_scores),
            "avg_precision": np.mean(all_precisions),
            "avg_recall": np.mean(all_recalls),
            "avg_f1": np.mean(all_f1_scores),
            "physics_consistency": np.mean(physics_consistencies) if physics_consistencies else 1.0,
            "noise_robustness": np.mean(noise_robustness_scores),
            "avg_inference_time": np.mean(inference_times),
            "peak_memory_mb": peak_memory
        }
    
    def _compute_classification_metrics(self, predicted: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 score."""
        try:
            # Flatten tensors
            pred_flat = predicted.flatten().cpu().numpy()
            target_flat = target.flatten().cpu().numpy()
            
            # Compute per-class metrics
            precisions = []
            recalls = []
            f1_scores = []
            
            for class_id in range(self.model_config["n_classes"]):
                # True positives, false positives, false negatives
                tp = np.sum((pred_flat == class_id) & (target_flat == class_id))
                fp = np.sum((pred_flat == class_id) & (target_flat != class_id))
                fn = np.sum((pred_flat != class_id) & (target_flat == class_id))
                
                # Precision and recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            
            return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1_scores))
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _evaluate_physics_consistency(self, outputs: torch.Tensor) -> float:
        """Evaluate physics consistency using Maxwell solver."""
        try:
            if self.maxwell_enabled:
                physics_constraint = self.maxwell_solver.solve(outputs)
                consistency = 1.0 / (1.0 + torch.mean(physics_constraint ** 2).item())
                return consistency
            return 1.0
        except Exception:
            return 1.0
    
    def _evaluate_noise_robustness(self, images: torch.Tensor, masks: torch.Tensor) -> float:
        """Evaluate model robustness to noise."""
        try:
            # Add noise and compare predictions
            noisy_images = quantum_noise_injection(images, factor=0.05)
            
            with torch.no_grad():
                clean_outputs = self.model(images)
                noisy_outputs = self.model(noisy_images)
                
                clean_pred = torch.argmax(clean_outputs, dim=1)
                noisy_pred = torch.argmax(noisy_outputs, dim=1)
                
                # Compute agreement between clean and noisy predictions
                agreement = (clean_pred == noisy_pred).float().mean().item()
                return agreement
                
        except Exception:
            return 1.0
    
    # 4. DATA HANDLING FUNCTIONS
    
    def _load_client_data(self):
        """Load client-specific data partition."""
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            
            # Check if data path exists
            if not self.data_path.exists():
                self.logger.warning(f"Data path does not exist: {self.data_path}, creating dummy data")
                self._create_minimal_dummy_data()
                return
            
            # Try to load ACDC dataset
            try:
                self.logger.info("Attempting to create ACDC dataloader...")
                # Create ACDC dataloader
                self.trainloader = create_acdc_dataloader(
                    data_dir=str(self.data_path),
                    batch_size=self.batch_size,
                    shuffle=True,
                    augment=True,
                    num_workers=0
                )
                
                # Create validation loader (subset of training data)
                self.valloader = create_acdc_dataloader(
                    data_dir=str(self.data_path),
                    batch_size=self.batch_size,
                    shuffle=False,
                    augment=False,
                    num_workers=0
                )
                
                # Count examples
                self.num_examples["trainset"] = len(self.trainloader.dataset)  # type: ignore
                self.num_examples["valset"] = len(self.valloader.dataset)  # type: ignore
                
                self.logger.info(f"Loaded ACDC data: {self.num_examples['trainset']} train, "
                               f"{self.num_examples['valset']} val samples")
                
            except Exception as e:
                self.logger.warning(f"Failed to load ACDC data: {e}")
                # Try alternative data loading
                try:
                    self._load_alternative_data()
                except Exception as e2:
                    self.logger.warning(f"Alternative data loading also failed: {e2}")
                    self._create_minimal_dummy_data()
                
        except Exception as e:
            self.logger.error(f"Failed to load client data: {e}")
            # Create minimal dummy data for testing
            self._create_minimal_dummy_data()
    
    def _load_alternative_data(self):
        """Load data using alternative methods."""
        try:
            # Look for image files in the directory
            image_files = []
            mask_files = []
            
            for ext in ['.nii', '.nii.gz']:
                image_files.extend(list(self.data_path.rglob(f'*{ext}')))
            
            # Filter out ground truth files
            image_files = [f for f in image_files if '_gt' not in f.name]
            
            # Find corresponding mask files
            for img_file in image_files:
                mask_file = img_file.parent / f"{img_file.stem}_gt{img_file.suffix}"
                if mask_file.exists():
                    mask_files.append(str(mask_file))
                else:
                    mask_files.append(None)
            
            if image_files:
                # Create dataloaders from file paths
                image_paths = [str(f) for f in image_files]
                
                self.trainloader = create_dataloader_from_paths(
                    image_paths=image_paths,
                    mask_paths=mask_files,
                    batch_size=self.batch_size,
                    shuffle=True,
                    augment=True
                )
                
                self.valloader = create_dataloader_from_paths(
                    image_paths=image_paths[:len(image_paths)//5],  # Use 20% for validation
                    mask_paths=mask_files[:len(mask_files)//5] if mask_files else None,
                    batch_size=self.batch_size,
                    shuffle=False,
                    augment=False
                )
                
                self.num_examples["trainset"] = len(image_paths)
                self.num_examples["valset"] = len(image_paths) // 5
                
                self.logger.info(f"Loaded alternative data: {self.num_examples['trainset']} train, "
                               f"{self.num_examples['valset']} val samples")
            else:
                raise ValueError("No image files found")
                
        except Exception as e:
            self.logger.error(f"Alternative data loading failed: {e}")
            raise
    
    def _create_minimal_dummy_data(self):
        """Create minimal dummy data for testing purposes."""
        self.logger.warning("Creating minimal dummy data - PERFORMANCE WILL BE POOR!")
        
        # Create small dummy dataset
        dummy_images = torch.randn(10, 1, MEDICAL_CONFIG['img_height'], MEDICAL_CONFIG['img_width'])
        dummy_masks = torch.randint(0, MEDICAL_CONFIG['num_classes'], 
                                  (10, MEDICAL_CONFIG['img_height'], MEDICAL_CONFIG['img_width']))
        
        dummy_dataset = torch.utils.data.TensorDataset(dummy_images, dummy_masks)
        
        self.trainloader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)
        self.valloader = DataLoader(dummy_dataset, batch_size=2, shuffle=False)
        
        self.num_examples["trainset"] = 10
        self.num_examples["valset"] = 10

# 5. CLIENT FACTORY FUNCTION

def client_fn(cid: str) -> Client:
    """
    Create and configure a FlowerClient instance.
    
    Args:
        cid: Client ID string
        
    Returns:
        Configured FlowerClient instance as Client
    """
    try:
        # 5.1 Client ID Processing
        # Parse client ID to get partition number
        partition_id = int(cid) if cid.isdigit() else 0
        num_partitions = 10  # Default number of partitions
        
        # 5.2 Environment Setup
        client_id = str(partition_id)
        
        # Set reproducible seed based on partition ID
        set_seed(42 + int(partition_id))
        
        # Setup logging
        logger = setup_federated_logger(
            client_id=client_id,
            log_dir="logs/clients",
            level=logging.INFO
        )
        
        logger.info(f"Creating client {client_id} (partition {partition_id}/{num_partitions})")
        
        # 5.3 Data Pipeline Initialization
        # Determine data path for this client
        base_data_path = "data/raw/ACDC"  # Default data path
        client_data_path = f"{base_data_path}/client_{partition_id}"
        
        # If client-specific path doesn't exist, use base path
        if not os.path.exists(client_data_path):
            client_data_path = base_data_path
            logger.info(f"Client-specific path not found, using base path: {client_data_path}")
        
        # 5.4 Model Configuration
        model_config_raw = {}  # Default empty config
        training_config_raw = {}  # Default empty config
        
        # Ensure configs are dictionaries
        model_config = model_config_raw if isinstance(model_config_raw, dict) else {}
        training_config = training_config_raw if isinstance(training_config_raw, dict) else {}
        
        # 5.5 Client Instance Creation
        client = FlowerClient(
            client_id=client_id,
            data_path=str(client_data_path),
            model_config=model_config,
            training_config=training_config,
            device=DEVICE
        )
        
        logger.info(f"Client {client_id} created successfully")
        return client.to_client()
    except Exception as e:
        logging.error(f"Failed to create client: {e}")
        raise

# 6. UTILITY FUNCTIONS

def memory_cleanup():
    """Clean up memory and cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def optimize_memory():
    """Optimize memory usage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory fraction if needed
        # torch.cuda.set_per_process_memory_fraction(0.8)

def monitor_resources() -> Dict[str, float]:
    """Monitor resource usage."""
    resources = {}
    
    if torch.cuda.is_available():
        resources["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        resources["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    
    # CPU and system memory monitoring could be added here
    return resources

def handle_training_error(error: Exception, client_id: str) -> Dict[str, Any]:
    """Handle training errors gracefully."""
    logging.error(f"Training error in client {client_id}: {error}")
    return {
        "error": str(error),
        "error_type": type(error).__name__,
        "client_id": client_id,
        "timestamp": time.time()
    }

# 7. CLIENTAPP CREATION AND EXPORT

# Create the ClientApp with comprehensive error handling
try:
    app = ClientApp(client_fn=client_fn)
    logging.info("ClientApp created successfully")
except Exception as e:
    logging.error(f"Failed to create ClientApp: {e}")
    raise

# Export for Flower framework
__all__ = ["app", "FlowerClient", "client_fn"]

# Version and compatibility information
__version__ = "1.0.0"
__flower_version__ = "1.8.0"
__description__ = "Comprehensive Federated Learning Client for Medical Image Segmentation" 
