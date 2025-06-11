# 1. HEADER AND IMPORTS SECTION

# 1.1 Standard Library Imports
import os
import sys
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, cast
from collections import OrderedDict
import json
import pickle
import time
import gc
from pathlib import Path
import psutil

# 1.2 Scientific Computing Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

# Import nibabel properly with type handling
try:
    import nibabel as nib  # type: ignore
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

# 1.3 Flower Framework Imports
from flwr.client import ClientApp, NumPyClient, Client
from flwr.common import Context, Config, NDArrays, Scalar

# 1.4 Project-Specific Imports - DIRECT IMPORTS ONLY
from src.models.unet_model import RobustMedVFL_UNet, CombinedLoss
from src.data.data import SimpleMedicalDataset, create_simple_dataloader
from src.utils.seed import set_seed
from src.utils.logger import setup_federated_logger
from src.utils.metrics import evaluate_model_with_research_metrics, convert_metrics_for_fl_server

# 1.5 Model Components Imports - NO FALLBACKS
from src.models.unet_model import (
    quantum_noise_injection, 
    MaxwellSolver, 
    ePURE, 
    CombinedLoss,
    adaptive_spline_smoothing
)

logging.info("Advanced components imported successfully - using full model capabilities")

# 2. GLOBAL CONFIGURATION AND SETUP

# 2.1 Environment Setup
# Suppress gRPC warnings for clean logging
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '2')
warnings.filterwarnings('ignore', category=UserWarning)

# Global Device Configuration
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

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
    'local_epochs': 5,  # FIXED: Match config.toml (was 2)
    'batch_size': 8,  # FIXED: Match config.toml (was 2)
    'learning_rate': 0.0005,  # FIXED: Match config.toml (was 0.001)
    'weight_decay': 1e-4,  # FIXED: Match config.toml (was 1e-6)
    'gradient_clip_norm': 1.0  # FIXED: Conservative clipping for stability
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

# Configuration Constants (Defaults, can be overridden by server config)
DEFAULT_LOCAL_EPOCHS = int(os.environ.get("DEFAULT_LOCAL_EPOCHS", "10")) 
DEFAULT_LEARNING_RATE = float(os.environ.get("DEFAULT_LEARNING_RATE", "0.003")) 
DEFAULT_BATCH_SIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", "2")) # Back to 2 as it worked before
DEFAULT_VALIDATION_SPLIT = float(os.environ.get("DEFAULT_VALIDATION_SPLIT", "0.2"))
DEFAULT_PATIENCE_EPOCHS = int(os.environ.get("DEFAULT_PATIENCE_EPOCHS", "5")) # Early stopping patience

# 3. FLOWERCLIENT CLASS DEFINITION

# Utility functions for model information
def get_model_info(model):
    """Get comprehensive information about a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'device': next(model.parameters()).device.type if total_params > 0 else 'cpu'
    }

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
        
        #  NEW: Initialize Flower-specific attributes to fix linter errors
        self._flower_context: Optional[Context] = None
        self._partition_id: Optional[int] = None
        
        # 3.1.2 Model Component Configuration
        self._initialize_model()
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
        
        # Add missing attributes for standard components
        self.quantum_noise_enabled = False
        self.maxwell_enabled = False
        self.epure_enabled = False
        self.noise_factor = 0.01
        
        # Initialize data immediately
        self._load_client_data()
        
        # OPTIMIZED: Single compact initialization line for research
        print(f" Client {client_id} ready")
        
        self.logger.debug(f"Client {client_id} initialized successfully")
    
    def _initialize_model(self):
        """Initialize the main model with all components."""
        try:
            self.model = RobustMedVFL_UNet(
                n_channels=self.model_config['n_channels'],
                n_classes=self.model_config['n_classes'],
            ).to(self.device)
            
            # Ensure model is in training mode and parameters are trainable
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Verify trainable parameters
            num_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
            if num_trainable == 0:
                self.logger.error("CRITICAL: No trainable parameters found!")
                for param in self.model.parameters():
                    param.requires_grad_(True)
            
            # Get model information
            self.model_info = get_model_info(self.model)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _initialize_data_components(self):
        """Initialize data preprocessing and augmentation components."""
        # FIXED: Remove unused preprocessor and augmentation since we handle preprocessing 
        # directly in SimpleACDCDataset.__getitem__ for better control and consistency
        
        # Initialize data loaders as None (will be set when data is loaded)
        self.trainloader = None
        self.valloader = None
        self.num_examples = {"trainset": 0, "valset": 0}
        
        # Add missing attributes
        self.batch_size = self.training_config.get('batch_size', DEFAULT_BATCH_SIZE)
        self.criterion = self._create_criterion()
        
        self.logger.info("Data components initialized")
    
    def _load_client_data(self):
        """Load client data using the simplified dataset system"""
        try:
            # Auto-detect number of supernodes from environment, config, or default
            num_supernodes = self._detect_num_supernodes()
            self.logger.info(f"Auto-detected supernodes: {num_supernodes}")
            
            # Extract partition ID dynamically  
            partition_id = self._extract_partition_id(num_supernodes)
            
            # Use SimpleMedicalDataset for efficiency
            self._load_acdc_with_simple_dataset(partition_id, num_supernodes)
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise RuntimeError(f"Client {self.client_id} failed to load data: {e}. Check data path: {self.data_path}")

    def _detect_num_supernodes(self) -> int:
        """Auto-detect number of supernodes from multiple sources (same logic as server)."""
        num_supernodes = None
        
        # DEBUG: Log context information
        if hasattr(self, '_flower_context') and self._flower_context:
            context = self._flower_context
            self.logger.info(f"DEBUG: _flower_context exists: {type(context)}")
            if hasattr(context, 'run_config'):
                self.logger.info(f"DEBUG: run_config exists: {type(context.run_config)}")
                if context.run_config:
                    # Try to see all attributes in run_config
                    try:
                        attrs = dir(context.run_config)
                        self.logger.info(f"DEBUG: run_config attributes: {[attr for attr in attrs if not attr.startswith('_')]}")
                    except:
                        pass
            if hasattr(context, 'state'):
                self.logger.info(f"DEBUG: context.state exists: {type(context.state)}")
        else:
            self.logger.warning("DEBUG: No _flower_context found")
        
        # 1. Try to get from Flower context run_config (PRIMARY SOURCE)
        if hasattr(self, '_flower_context') and self._flower_context:
            context = self._flower_context
            if hasattr(context, 'run_config') and context.run_config:
                # Try direct attribute first
                raw_num_supernodes = getattr(context.run_config, 'num_supernodes', None)
                if raw_num_supernodes:
                    try:
                        num_supernodes = int(str(raw_num_supernodes))
                        self.logger.info(f"Detected num_supernodes from run_config: {num_supernodes}")
                    except (ValueError, TypeError):
                        pass
                
                # Try options.num-supernodes format (from pyproject.toml)
                if not num_supernodes and hasattr(context.run_config, 'options'):
                    options = getattr(context.run_config, 'options')
                    if options:
                        # Try different possible attribute names
                        for attr_name in ['num_supernodes', 'num-supernodes']:
                            raw_num_supernodes = getattr(options, attr_name, None)
                            if raw_num_supernodes:
                                try:
                                    num_supernodes = int(str(raw_num_supernodes))
                                    self.logger.info(f"Detected num_supernodes from run_config.options.{attr_name}: {num_supernodes}")
                                    break
                                except (ValueError, TypeError):
                                    continue
        
        # 2. Try to get from context state
        if not num_supernodes and hasattr(self, '_flower_context') and self._flower_context:
            context = self._flower_context
            if hasattr(context, 'state') and context.state:
                raw_num_supernodes = context.state.get('num_supernodes')
                if raw_num_supernodes:
                    try:
                        num_supernodes = int(str(raw_num_supernodes))
                        self.logger.info(f"Detected num_supernodes from context state: {num_supernodes}")
                    except (ValueError, TypeError):
                        pass
        
        # 3. Check environment variables
        if not num_supernodes:
            env_vars = ['FLWR_NUM_SUPERNODES', 'NUM_SUPERNODES', 'FLOWER_NUM_SUPERNODES']
            for env_var in env_vars:
                value = os.environ.get(env_var)
                if value:
                    try:
                        num_supernodes = int(value)
                        self.logger.info(f"Detected num_supernodes from env var {env_var}: {num_supernodes}")
                        break
                    except ValueError:
                        continue
        
        # 4. Try to parse from sys.argv (command line arguments)
        if not num_supernodes:
            import sys
            try:
                argv = sys.argv
                for i, arg in enumerate(argv):
                    if arg == '--num-supernodes' and i + 1 < len(argv):
                        num_supernodes = int(argv[i + 1])
                        self.logger.info(f"Detected num_supernodes from command line: {num_supernodes}")
                        break
            except (ValueError, IndexError):
                pass
        
        # 5. Try to auto-detect from pyproject.toml federation configs
        if not num_supernodes:
            try:
                import toml
                pyproject_path = Path('pyproject.toml')
                if pyproject_path.exists():
                    with open(pyproject_path, 'r') as f:
                        pyproject_config = toml.load(f)
                    
                    # Look for federation configs
                    federations = pyproject_config.get('tool', {}).get('flwr', {}).get('federations', {})
                    
                    # Try production federation first, then others
                    for federation_name in ['production', 'scalable', 'research', 'default']:
                        if federation_name in federations:
                            federation_config = federations[federation_name]
                            
                            # Try nested options structure first
                            if 'options' in federation_config:
                                options = federation_config['options']
                                for num_supernodes_key in ['num-supernodes', 'num_supernodes']:
                                    if num_supernodes_key in options:
                                        try:
                                            num_supernodes = int(options[num_supernodes_key])
                                            self.logger.info(f"Detected num_supernodes from pyproject.toml federation '{federation_name}': {num_supernodes}")
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                if num_supernodes:
                                    break
                            
                            # Also try flat structure as fallback
                            elif 'options.num-supernodes' in federation_config:
                                try:
                                    num_supernodes = int(federation_config['options.num-supernodes'])
                                    self.logger.info(f"Detected num_supernodes from pyproject.toml federation '{federation_name}' (flat): {num_supernodes}")
                                    break
                                except (ValueError, TypeError):
                                    continue
            except Exception as e:
                self.logger.debug(f"Failed to read pyproject.toml: {e}")
        
        # 6. Use default only if nothing found (should not happen in simulation)
        if num_supernodes is None:
            num_supernodes = 5
            self.logger.warning(f"No num_supernodes found, using default: {num_supernodes}")
        
        return num_supernodes

    def _load_acdc_with_simple_dataset(self, partition_id: int, num_supernodes: int):
        """Load ACDC data using the simplified dataset system"""
        self.logger.info(f"Loading ACDC with SimpleMedicalDataset: {num_supernodes} supernodes, partition {partition_id}")
        
        # Get ACDC data directory
        acdc_data_dir = self._get_acdc_data_dir()
        
        # Create simple dataloaders with built-in partitioning
        self.trainloader = create_simple_dataloader(
            data_dir=acdc_data_dir,
            batch_size=self.batch_size,
            shuffle=True,
            augment=True,  # Training with augmentation
            num_workers=0,
            client_id=partition_id,
            num_clients=num_supernodes,
            cache_size=50  # Cache 50 samples in memory
        )
        
        self.valloader = create_simple_dataloader(
            data_dir=acdc_data_dir,
            batch_size=self.batch_size,
            shuffle=False,
            augment=False,  # Validation without augmentation
            num_workers=0,
            client_id=partition_id,
            num_clients=num_supernodes,
            cache_size=20  # Smaller cache for validation
        )
        
        # Calculate number of examples with safe dataset size detection
        def get_dataset_size(dataloader):
            """Safely get dataset size with fallback methods."""
            if dataloader is None:
                return 0
            try:
                return len(dataloader.dataset)
            except (TypeError, AttributeError):
                # Fallback: estimate from dataloader
                try:
                    return len(dataloader) * dataloader.batch_size
                except:
                    return 0
        
        self.num_examples = {
            "trainset": get_dataset_size(self.trainloader), 
            "valset": get_dataset_size(self.valloader)
        }
        
        # Verify data is not empty
        if self.num_examples["trainset"] == 0:
            raise ValueError(f"No training data found for client {self.client_id}")
        
        self.logger.info(f"ACDC dataset loaded successfully:")
        self.logger.info(f"Training: {self.num_examples['trainset']} samples")
        self.logger.info(f"Validation: {self.num_examples['valset']} samples") 
        self.logger.info(f"Client: {partition_id}/{num_supernodes}")
        
        # Store simple metadata
        self.research_metadata = {
            'client_id': partition_id,
            'num_supernodes': num_supernodes,
            'trainset_size': self.num_examples["trainset"],
            'valset_size': self.num_examples["valset"]
        }

    def _extract_partition_id(self, num_supernodes: int) -> int:
        """Extract partition ID from client_id with multiple strategies"""
        
        # 1. If already set by Flower context
        if hasattr(self, '_partition_id') and self._partition_id is not None:
            return self._partition_id
        
        # 2. Extract from numeric part of client_id
        import re
        numeric_match = re.search(r'\d+', self.client_id)
        if numeric_match:
            partition_id = int(numeric_match.group()) % num_supernodes
            self.logger.info(f"Extracted partition_id {partition_id} from client_id '{self.client_id}'")
            return partition_id
        
        # 3. Use hash for non-numeric client IDs
        partition_id = abs(hash(self.client_id)) % num_supernodes
        self.logger.info(f"Generated partition_id {partition_id} from hash of client_id '{self.client_id}'")
        return partition_id

    def _get_acdc_data_dir(self) -> str:
        """Get ACDC data directory with fallback paths"""
        # First try the correct path from project structure
        acdc_data_dir = os.path.join(self.data_path, "database", "training")
        if os.path.exists(acdc_data_dir):
            return acdc_data_dir
            
        # Fallback paths for different configurations
        fallback_paths = [
            os.path.join(self.data_path, "training"),  # Legacy path
            str(self.data_path)  # Direct path
        ]
        
        for fallback_path in fallback_paths:
            if os.path.exists(fallback_path):
                return fallback_path
                
        raise FileNotFoundError(f"ACDC training data not found. Tried paths: {acdc_data_dir}, {fallback_paths}")

    def _create_criterion(self):
        """Create loss function for training with enhanced class balancing for medical data."""
        try:
            # FULL COMBINED LOSS: Enable all components for comprehensive training
            print(f"[Client {self.client_id}] Using FULL CombinedLoss with ALL components enabled (CE, Dice, Physics, Smoothness)")
            
            # Enhanced class weights for medical segmentation
            class_weights = torch.tensor([0.1, 2.0, 2.5, 1.5], dtype=torch.float32, device=self.device)  # [Background, RV, Myocardium, LV]
            
            # Use CombinedLoss with ALL components enabled
            num_model_classes = self.model_config.get('n_classes', 4)
            # The bottleneck of RobustMedVFL_UNet outputs 1024 channels
            bottleneck_channels = 1024 # Should match RobustMedVFL_UNet's bottleneck_features channels

            criterion = CombinedLoss(
                wc=0.5,  # CrossEntropy weight
                wd=0.5,  # Dice weight
                wp=0.1,  # ENABLED Physics weight
                ws=0.01, # ENABLED Smoothness weight
                in_channels_maxwell=bottleneck_channels, # Channels for b1_map_placeholder (bottleneck_features)
                num_classes=num_model_classes,
                max_kappa=300.0
            )
            
            # CRITICAL: Override the CrossEntropy component with class weights
            if hasattr(criterion, 'ce_loss'):
                criterion.ce_loss = nn.CrossEntropyLoss(
                    weight=class_weights, 
                    reduction='mean',
                    label_smoothing=0.1  # Add label smoothing for regularization
                )
                print(f"Enhanced CrossEntropy loss with class weights applied")
            
            return criterion
                
        except Exception as e:
            self.logger.error(f"Failed to create CombinedLoss: {e}")
            
            # EMERGENCY FALLBACK: Weighted CrossEntropyLoss only
            print(f"Falling back to weighted CrossEntropy only")
            class_weights = torch.tensor([0.1, 2.0, 3.0, 2.0], dtype=torch.float32, device=self.device)
            fallback_criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                reduction='mean',
                label_smoothing=0.1
            )
            
            return fallback_criterion
    
    # 3.2 PARAMETER MANAGEMENT METHODS
    
    def get_parameters(self, config: Config) -> NDArrays:
        """Extract model parameters for federated aggregation."""
        try:
            parameters = []
            trainable_count = 0
            total_count = 0
            
            for param in self.model.parameters():
                total_count += 1
                if param.requires_grad:
                    param_array = param.detach().cpu().numpy()
                    parameters.append(param_array)
                    trainable_count += 1
            
            # If no trainable parameters found, use all parameters
            if len(parameters) == 0:
                self.logger.warning("No trainable parameters found! Using all parameters as fallback")
                for param in self.model.parameters():
                    param_array = param.detach().cpu().numpy()
                    parameters.append(param_array)
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Error extracting parameters: {e}")
            raise
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from federated aggregation."""
        try:
            # Use same strategy as get_parameters
            model_params = list(self.model.parameters())
            trainable_params = [p for p in model_params if p.requires_grad]
            
            # If no trainable parameters found, use all parameters
            if len(trainable_params) == 0:
                self.logger.warning("No trainable parameters found in set_parameters! Using all parameters as fallback")
                target_params = model_params
            else:
                target_params = trainable_params
            
            # Validate parameter count
            if len(parameters) != len(target_params):
                raise ValueError(f"Parameter count mismatch: expected {len(target_params)}, "
                               f"got {len(parameters)}")
            
            # Set parameters directly to model.parameters()
            with torch.no_grad():
                for param_tensor, new_param_array in zip(target_params, parameters):
                    new_param_tensor = torch.from_numpy(new_param_array).to(self.device)
                    
                    if new_param_tensor.shape != param_tensor.shape:
                        raise ValueError(f"Shape mismatch: expected {param_tensor.shape}, "
                                       f"got {new_param_tensor.shape}")
                    
                    param_tensor.copy_(new_param_tensor)
            
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
        Train model with provided parameters.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
            
        Returns:
            Updated parameters, number of examples, and training metrics
        """
        start_time = time.time()
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Extract training configuration with optimized defaults
        local_epochs = int(config.get("local_epochs", 5))  # Increased from 1 to 5
        batch_size = int(config.get("batch_size", 4))  # Optimal batch size for medical data
        learning_rate = float(config.get("learning_rate", 5e-4))  # Reduced for stability
        weight_decay = float(config.get("weight_decay", 1e-4))  # Increased regularization
        
        # Silent initialization - remove verbose logging
        
        # Initialize scheduler to None first
        scheduler = None
        
        # Update optimizer learning rate if changed
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        else:
            # Create optimizer if not exists with OPTIMIZED settings for HIGH PERFORMANCE
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,  # Increased regularization
                betas=(0.9, 0.999),  # Standard Adam betas for stability
                eps=1e-8  # Standard Adam epsilon
            )
            
            # ENABLED: Learning rate scheduler for better convergence like centralized
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=local_epochs,  # Cosine annealing over local epochs
                eta_min=learning_rate * 0.1  # Minimum LR = 10% of initial
            )
        
        # Ensure trainloader is available
        if self.trainloader is None:
            self._load_client_data()
        
        if self.trainloader is None:
            # Return valid parameters with at least 1 example to prevent division by zero
            return self.get_parameters(config), 1, {"error": "No training data", "train_loss": float('inf')}
        
        # Silent training loop with minimal logging
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        num_examples = 0
        batch_losses = []
        
        # ADDED: Data validation and debugging
        first_batch_logged = False
        
        # Early stopping variables
        best_epoch_loss = float('inf')
        patience_counter = 0
        patience_limit = 3  # Stop if no improvement for 3 epochs
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, (images, masks) in enumerate(self.trainloader):
                # Move to device
                images = images.to(self.device).float()
                masks = masks.to(self.device).long()
                
                # Data validation for first batch
                if not first_batch_logged:
                    unique_masks = torch.unique(masks).cpu().numpy()
                    validation_issues = []
                    
                    # Check for invalid classes
                    if not all(c in [0, 1, 2, 3] for c in unique_masks):
                        validation_issues.append(f"INVALID CLASSES: {unique_masks}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(images).any() or torch.isinf(images).any():
                        validation_issues.append("IMAGE NaN/Inf detected")
                    
                    if torch.isnan(masks.float()).any():
                        validation_issues.append("MASK NaN detected")
                    
                    # Report validation results
                    if validation_issues:
                        print(f"[Client {self.client_id}] Data issues: {', '.join(validation_issues)}")
                    
                    first_batch_logged = True
                
                # Apply quantum noise injection for robustness (with lower probability)
                if self.quantum_noise_enabled and torch.rand(1).item() < 0.3:  # 30% chance
                    images = quantum_noise_injection(images)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                # Handle physics-informed model outputs: (logits, bottleneck_features, all_eps_sigma_tuples)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    logits, bottleneck_features, eps_sigma_list = outputs
                elif isinstance(outputs, tuple) and len(outputs) == 1: # Should not happen with RobustMedVFL_UNet
                    logits = outputs[0]
                    bottleneck_features = None
                    eps_sigma_list = None
                    if epoch == 0 and batch_idx == 0:
                        self.logger.warning("Model returned a single tuple element, expected 3 for physics features. Physics/Smoothness loss might be ineffective.")
                else: # Standard model output
                    logits = outputs
                    bottleneck_features = None
                    eps_sigma_list = None
                    if epoch == 0 and batch_idx == 0:
                        self.logger.warning("Model did not return a tuple, expected 3 for physics features. Physics/Smoothness loss might be ineffective.")

                # Ensure logits is a tensor for type safety
                if not torch.is_tensor(logits):
                    self.logger.warning(f"Model output is not a tensor: {type(logits)}")
                    continue
                
                # Validate logits for numerical issues
                if batch_idx == 0 and epoch == 0:
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"[Client {self.client_id}] ERROR: NaN/Inf in logits detected")
                    
                    #  CRITICAL: Softmax distribution analysis (detect model collapse)
                    logits_softmax = F.softmax(logits, dim=1)
                    print(f"     Softmax range: [{logits_softmax.min():.6f}, {logits_softmax.max():.6f}]")
                    
                    # Check each class probability distribution
                    print(f"     Per-class softmax analysis:")
                    for class_idx in range(4):
                        class_probs = logits_softmax[:, class_idx, :, :]
                        class_mean = class_probs.mean().item()
                        class_max = class_probs.max().item()
                        print(f"       Class {class_idx}: Mean={class_mean:.4f}, Max={class_max:.4f}")
                        
                        #  Detect model collapse for each class
                        if class_mean > 0.8:
                            print(f"        COLLAPSE: Class {class_idx} dominates (mean={class_mean:.3f})!")
                        elif class_mean < 0.05:
                            print(f"         WEAK: Class {class_idx} barely predicted (mean={class_mean:.3f})")
                    
                    #  CRITICAL: Prediction analysis
                    predictions = torch.argmax(logits_softmax, dim=1)
                    pred_unique = torch.unique(predictions).cpu().numpy()
                    print(f"     Predicted classes: {pred_unique}")
                    
                    # Check for prediction collapse (model always predicts same class)
                    if len(pred_unique) == 1:
                        print(f"      PREDICTION COLLAPSE: Model only predicts class {pred_unique[0]}!")
                    
                    # Prediction distribution
                    total_pred_pixels = predictions.numel()
                    print(f"     Prediction distribution:")
                    for class_id in range(4):
                        pred_count = (predictions == class_id).sum().item()
                        pred_ratio = pred_count / total_pred_pixels
                        print(f"       Pred Class {class_id}: {pred_count:6d} pixels ({pred_ratio:.3%})")
                    
                    #  CRITICAL: Compare predictions vs ground truth
                    print(f"     Ground truth vs Predictions comparison:")
                    for class_id in range(4):
                        gt_count = (masks == class_id).sum().item()
                        pred_count = (predictions == class_id).sum().item()
                        gt_ratio = gt_count / total_pred_pixels
                        pred_ratio = pred_count / total_pred_pixels
                        diff = abs(gt_ratio - pred_ratio)
                        print(f"       Class {class_id}: GT={gt_ratio:.3%} vs Pred={pred_ratio:.3%} (diff={diff:.3%})")
                        
                        if diff > 0.5:  # >50% difference
                            print(f"          HUGE MISMATCH: >50% difference for class {class_id}!")
                    
                    # Validate expected number of classes
                    if logits.shape[1] != 4:
                        print(f"      ERROR: Expected 4 classes, got {logits.shape[1]}")
                
                # Compute loss - Using CombinedLoss from unet_model
                # CombinedLoss accepts: logits, targets, b1_map_placeholder, all_eps_sigma_tuples, features_for_smoothness
                loss_result = self.criterion(
                    logits, 
                    masks, 
                    b1_map_placeholder=bottleneck_features, # Pass actual bottleneck features
                    all_eps_sigma_tuples=eps_sigma_list,    # Pass actual eps_sigma_list
                    features_for_smoothness=bottleneck_features # Use bottleneck_features for smoothness loss
                )
                
                # CombinedLoss returns scalar tensor
                if torch.is_tensor(loss_result):
                    loss = loss_result
                    
                    # ENHANCED: Detailed loss validation and debugging
                    if batch_idx == 0 and epoch == 0:
                        print(f"   Loss Computation:")
                        print(f"     Raw loss: {loss.item():.6f}")
                        print(f"     Loss dtype: {loss.dtype}")
                        print(f"     Loss device: {loss.device}")
                        
                        # Verify loss is in reasonable range
                        if loss.item() > 100:
                            print(f"      CATASTROPHIC LOSS DETECTED: {loss.item():.2e}")
                            print(f"      This indicates severe numerical instability!")
                    
                    # Check for problematic loss values - ensure tensor input
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print(f"      ERROR: NaN/Inf loss detected: {loss.item()}")
                        continue
                            
                else:
                    raise ValueError(f"Unexpected loss type: {type(loss_result)}")
                
                # Backward pass with ENHANCED gradient clipping for FL stability
                loss.backward()
                
                #  ENHANCED: Comprehensive gradient flow analysis
                total_grad_norm = 0.0
                param_count = 0
                zero_grad_count = 0
                max_grad_norm = 0.0
                min_grad_norm = float('inf')
                grad_norms = []
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                        param_count += 1
                        grad_norms.append(param_norm.item())
                        max_grad_norm = max(max_grad_norm, param_norm.item())
                        min_grad_norm = min(min_grad_norm, param_norm.item())
                        
                        if param_norm.item() < 1e-8:
                            zero_grad_count += 1
                            
                        # Log problematic gradients for first few parameters
                        if batch_idx == 0 and epoch == 0 and param_count <= 3:
                            print(f"     Grad[{name[:20]}]: {param_norm.item():.8f}")
                    else:
                        zero_grad_count += 1
                        
                total_grad_norm = total_grad_norm ** 0.5
                
                # Comprehensive gradient analysis for first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"   Gradient Flow Analysis:")
                    print(f"     Total grad norm: {total_grad_norm:.8f}")
                    print(f"     Max grad norm: {max_grad_norm:.8f}")
                    print(f"     Min grad norm: {min_grad_norm:.8f}")
                    print(f"     Zero gradients: {zero_grad_count}/{param_count + zero_grad_count}")
                    print(f"     Mean grad norm: {np.mean(grad_norms):.8f}")
                    print(f"     Std grad norm: {np.std(grad_norms):.8f}")
                    
                    #  Gradient health checks
                    gradient_issues = []
                    if total_grad_norm < 1e-6:
                        gradient_issues.append("Extremely small gradients - model may not be learning!")
                    if total_grad_norm > 100:
                        gradient_issues.append("Extremely large gradients - may cause instability!")
                    if zero_grad_count > param_count * 0.5:
                        gradient_issues.append(f"Too many zero gradients ({zero_grad_count}/{param_count})")
                    
                    if gradient_issues:
                        print(f"      GRADIENT ISSUES:")
                        for issue in gradient_issues:
                            print(f"          {issue}")
                    else:
                        print(f"      Gradient flow appears healthy")
                
                # ENHANCED gradient clipping based on gradient analysis
                if total_grad_norm > 10.0:
                    # Aggressive clipping for very large gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if batch_idx == 0 and epoch == 0:
                        print(f"      Applied aggressive gradient clipping (norm {total_grad_norm:.2f} -> 1.0)")
                elif total_grad_norm > 5.0:
                    # Moderate clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                    if batch_idx == 0 and epoch == 0:
                        print(f"      Applied moderate gradient clipping (norm {total_grad_norm:.2f} -> 2.0)")
                else:
                    # Light clipping for normal gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                #  ENHANCED: Post-optimization parameter changes
                if batch_idx == 0 and epoch == 0:
                    print(f"   Parameter Update Analysis:")
                    param_changes = []
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            # This would need parameter storage from before update to compute actual change
                            param_changes.append(param.data.norm().item())
                    
                    if param_changes:
                        mean_param_norm = np.mean(param_changes)
                        print(f"     Mean parameter norm after update: {mean_param_norm:.8f}")
                        if mean_param_norm < 1e-6 or mean_param_norm > 100:
                            print(f"      WARNING: Unusual parameter magnitude!")
                    
                    print(f"   First batch analysis complete\n")
                
                # Accumulate metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss
                epoch_batches += 1
                num_examples += images.size(0)
                batch_losses.append(batch_loss)
            
            # Step learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Early stopping check
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else float('inf')
            if avg_epoch_loss < best_epoch_loss:
                best_epoch_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"[C{self.client_id}] Early stopping at epoch {epoch+1}/{local_epochs}")
                    break
            
            total_loss += epoch_loss
            num_batches += epoch_batches
        
        # Calculate final metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        training_time = time.time() - start_time
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config)
        
        # Prepare comprehensive metrics for AdaFedAdam server - SIMPLIFIED FOR SERIALIZATION
        metrics = {
            "train_loss": float(avg_loss),
            "num_examples": int(num_examples),
            "num_batches": int(num_batches),
            "training_time": float(training_time),
            "local_epochs": int(local_epochs),
            "final_lr": float(learning_rate),
            
            # AdaFedAdam-specific metrics for fairness weighting
            "initial_loss": float(batch_losses[0]) if batch_losses else float(avg_loss),
            "final_loss": float(batch_losses[-1]) if batch_losses else float(avg_loss),
            "loss_improvement": float(batch_losses[0] - batch_losses[-1]) if len(batch_losses) > 1 else 0.0,
            "loss_variance": float(np.var(batch_losses)) if len(batch_losses) > 1 else 0.0,
            "convergence_rate": float(abs(batch_losses[0] - batch_losses[-1]) / max(batch_losses[0], 1e-6)) if len(batch_losses) > 1 else 0.0,
            
            # Client-specific information for server monitoring
            "client_data_quality": float(1.0 - min(avg_loss, 2.0) / 2.0),
            "training_stability": float(1.0 / (1.0 + float(np.std(batch_losses)))) if len(batch_losses) > 1 else 1.0,
        }
        
        # Add κ (kappa) values from AdaptiveTvMFDiceLoss - SIMPLIFIED
        try:
            if hasattr(self.criterion, 'get_kappa_values'):
                kappa_attr = getattr(self.criterion, 'get_kappa_values')
                if callable(kappa_attr):
                    kappa_values = kappa_attr()
                else:
                    kappa_values = kappa_attr
                    
                if kappa_values and isinstance(kappa_values, dict):
                    # Only add simple scalar kappa values
                    for key, value in kappa_values.items():
                        if isinstance(value, (int, float, np.number)):
                            metrics[f"kappa_{key}"] = float(value)
        except Exception:
            pass  # Ignore kappa extraction errors
        
        # OPTIMIZED: Single compact research metrics line for client completion
        loss_improvement = metrics['loss_improvement']
        stability = metrics['training_stability']
        data_quality = metrics['client_data_quality']
        convergence = metrics['convergence_rate']
        
        # COMPACT CLIENT FORMAT
        print(f"[C{self.client_id}] R{config.get('server_round', 0):02d} | Loss={avg_loss:.4f} | Δ={loss_improvement:.3f} | Examples={num_examples} | Time={training_time:.1f}s")
        
        return updated_parameters, int(self.num_examples["trainset"]), metrics

    # 3.4 EVALUATION METHOD (EVALUATE) - COMPREHENSIVE IMPLEMENTATION
    
    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model with COMPREHENSIVE PER-CLIENT METRICS
        
        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration from server
            
        Returns:
            Loss value, number of examples, and comprehensive evaluation metrics
        """
        start_time = time.time()
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Extract evaluation configuration
        batch_size = int(config.get("batch_size", 4))
        enable_detailed_metrics = config.get("enable_detailed_metrics", True)
        
        # Ensure validation loader is available
        if self.valloader is None:
            self._load_client_data()
        
        if self.valloader is None:
            # Return minimal valid response to prevent server crash
            return 1.0, 1, {"error": "No validation data", "eval_loss": 1.0}
        
        # USE UTILS/METRICS.PY FOR EVALUATION INSTEAD OF MANUAL IMPLEMENTATION
        self.logger.info("Starting comprehensive evaluation using utils/metrics.py")
        
        # Use the research-grade metrics evaluation from utils/metrics.py
        research_metrics = evaluate_model_with_research_metrics(
            model=self.model,
            dataloader=self.valloader,
            device=self.device,
            num_classes=4,
            class_names=['Background', 'RV', 'Myocardium', 'LV'],
            return_detailed=False
        )
        
        # Convert to FL-compatible format
        fl_compatible_metrics = convert_metrics_for_fl_server(research_metrics)
        
        # Extract essential values
        avg_loss = fl_compatible_metrics.get('eval_loss', 1.0)
        num_examples = int(fl_compatible_metrics.get('num_examples', 0))
        evaluation_time = time.time() - start_time
        
        # CREATE CLIENT METRICS FROM RESEARCH METRICS
        metrics = {
            "eval_loss": float(avg_loss),
            "num_examples": int(num_examples),
            "num_batches": int(research_metrics.get('total_batches', 0)),
            "evaluation_time": float(evaluation_time),
            "client_id": str(self.client_id),
        }
        
        # Add all FL-compatible metrics
        metrics.update(fl_compatible_metrics)
        
        # EXTRACT VALUES FOR PRINTING (from research_metrics for accuracy)
        mean_dice = research_metrics.get('mean_dice_fg', 0.0)  # Foreground dice
        mean_iou = research_metrics.get('mean_iou_fg', 0.0)    # Foreground IoU
        mean_precision = research_metrics.get('mean_precision', 0.0)
        mean_recall = research_metrics.get('mean_recall', 0.0)
        mean_f1 = research_metrics.get('mean_f1', 0.0)
        
        dice_scores = research_metrics.get('dice_scores', [0.0, 0.0, 0.0, 0.0])
        iou_scores = research_metrics.get('iou_scores', [0.0, 0.0, 0.0, 0.0])
        precision_scores = research_metrics.get('precision_scores', [0.0, 0.0, 0.0, 0.0])
        recall_scores = research_metrics.get('recall_scores', [0.0, 0.0, 0.0, 0.0])
        f1_scores = research_metrics.get('f1_scores', [0.0, 0.0, 0.0, 0.0])
        
        # ADAPTIVE KAPPA: Extract dice scores for potential κ adaptation
        try:
            metrics.update({
                'kappa_enabled': True,
                'dice_for_kappa': dice_scores,  # Use research metrics dice scores
            })
        except Exception as e:
            self.logger.warning(f"Kappa metrics collection failed: {e}")
        
        # DETAILED FEDERATED LEARNING ROUND RESULTS (like training epochs)
        current_round = config.get("round", "Unknown")
        print(f"\n" + "="*80)
        print(f"--- FL Round {current_round} Evaluation Results ---")
        print(f"   Round {current_round} - Evaluation Loss: {avg_loss:.4f}")
        print(f"   Evaluating on validation set...")
        print(f"   Round {current_round} - Validation (Avg Foreground): Dice: {mean_dice:.4f}; IoU: {mean_iou:.4f}; Precision: {mean_precision:.4f}; Recall: {mean_recall:.4f}; F1-score: {mean_f1:.4f}")
        
        # Per-class detailed metrics (same format as training epochs)
        class_labels = ["Background", "RV", "Myocardium", "LV"]
        for i, class_name in enumerate(class_labels):
            if i < len(dice_scores):
                dice = dice_scores[i]
                iou = iou_scores[i] if i < len(iou_scores) else 0.0
                precision = precision_scores[i] if i < len(precision_scores) else 0.0
                recall = recall_scores[i] if i < len(recall_scores) else 0.0
                f1 = f1_scores[i] if i < len(f1_scores) else 0.0
                
                print(f"     Class {i}: Dice: {dice:.4f}; IoU: {iou:.4f}; Precision: {precision:.4f}; Recall: {recall:.4f}; F1-score: {f1:.4f}")
        
        print(f"   Client: {self.client_id} | Data: {num_examples} samples | Time: {evaluation_time:.1f}s")
        print("="*80)
        
        # Brief summary for FL logs
        print(f"[Client {self.client_id}] Round {current_round} Summary: Loss={avg_loss:.4f} | Dice={mean_dice:.4f} | Data={num_examples}")
        
        return float(avg_loss), int(num_examples), metrics

# 4. CLIENT APP FACTORY FUNCTION - REQUIRED FOR FLOWER 1.18.0

def client_fn(context: Context) -> Client:
    """Create a Flower client instance with enhanced partition detection"""
    try:
        # Extract client configuration from context
        client_id = context.node_config.get("client-id", "default_client")
        data_path = context.node_config.get("data-path", "data/raw/ACDC")
        
        # Extract model and training configurations
        model_config = {
            'n_channels': context.node_config.get('n-channels', 1),
            'n_classes': context.node_config.get('n-classes', 4),
            'dropout_rate': context.node_config.get('dropout-rate', 0.1),
            'use_batch_norm': context.node_config.get('use-batch-norm', True),
            'use_residual': context.node_config.get('use-residual', True)
        }
        
        training_config = {
            'local_epochs': context.node_config.get('local-epochs', DEFAULT_LOCAL_EPOCHS),
            'batch_size': context.node_config.get('batch-size', DEFAULT_BATCH_SIZE),
            'learning_rate': context.node_config.get('learning-rate', DEFAULT_LEARNING_RATE),
            'weight_decay': context.node_config.get('weight-decay', 1e-5),
            'gradient_clip_norm': context.node_config.get('gradient-clip-norm', 1.0)
        }
        
        # Create client instance
        flower_client = FlowerClient(
            client_id=str(client_id),
            data_path=str(data_path),
            model_config=model_config,
            training_config=training_config,
            device=DEVICE
        )
        
        #  NEW: Enhanced Flower integration
        # Store Flower context for auto-detection
        flower_client._flower_context = context
        
        # Extract partition-id from Flower context (for simulation)
        partition_id = context.node_config.get("partition-id", None)
        if partition_id is not None:
            flower_client._partition_id = int(partition_id)
        
        flower_client.logger.info(f"Client {client_id} created with Flower partitioning support")
        
        #  FIXED: Convert NumPyClient to Client using the official method
        return flower_client.to_client()
        
    except Exception as e:
        # Fallback configuration if context fails
        print(f"Warning: Failed to extract config from context: {e}")
        print("Using default client configuration")
        
        flower_client = FlowerClient(
            client_id="default_client",
            data_path="data/raw/ACDC",
            model_config=DEFAULT_MODEL_CONFIG,
            training_config=DEFAULT_TRAINING_CONFIG,
            device=DEVICE
        )
        
        #  FIXED: Convert NumPyClient to Client using the official method
        return flower_client.to_client()

# 5. CLIENT APP DEFINITION - REQUIRED FOR FLOWER 1.18.0

app = ClientApp(client_fn=client_fn)

# 6. DEVELOPMENT AND TESTING SUPPORT

if __name__ == "__main__":

    # Direct testing option
    import sys
    
    if "--test-client" in sys.argv:
        try:
            print("\n Testing client initialization...")
            
            # Create a dummy context for testing
            from flwr.common import Context
            
            class DummyNodeConfig:
                def get(self, key, default=None):
                    config_map = {
                        'client-id': 'test_client_001',
                        'data-path': 'data/raw/ACDC',
                        'n-channels': 1,
                        'n-classes': 4,
                        'local-epochs': 2,
                        'batch-size': 2,
                        'learning-rate': 0.001
                    }
                    return config_map.get(key, default)
            
            class DummyContext:
                def __init__(self):
                    self.node_config = DummyNodeConfig()
            
            dummy_context = DummyContext()
            
            # Test client creation  
            test_client = client_fn(dummy_context)  # type: ignore
            
            print(f"Client created successfully")
            print(f"   Client type: {type(test_client).__name__}")
            print(f"   Device available")
            
            # Simple test without accessing specific attributes
            print(f"   Client initialized and ready for FL")
            
            print("\nCLIENT TEST COMPLETED SUCCESSFULLY!")
            print("The client is ready for federated learning deployment.")
            
        except Exception as e:
            print(f"\nCLIENT TEST FAILED: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("\n Use --test-client flag to test client initialization")
        print("For production deployment, use the CLI commands above.")


