# 1. HEADER AND IMPORTS SECTION

# 1.1 Standard Library Imports
import os
import sys
import logging
import warnings
import time
import json
import pickle
import math
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from collections import OrderedDict
from pathlib import Path

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
from flwr.server.strategy import FedAdam  # Using standard FedAdam
from flwr.common import Context, Parameters, FitRes, EvaluateRes
from flwr.common.typing import Scalar, Metrics, NDArrays
from flwr.common import FitIns, EvaluateIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

# 1.4 Project-Specific Imports - FIXED
try:
    from src.models.mlp_model import (
        RobustMedVFL_UNet,
        CombinedLoss,
        quantum_noise_injection
    )
    from src.data.dataset import ACDCDataset
    from src.data.loader import create_acdc_dataloader, create_dataloader
    from src.utils.seed import set_seed
    from src.utils.logger import setup_federated_logger
    from src.utils.metrics import evaluate_metrics, compute_class_weights, print_metrics_summary
except ImportError:
    # Fallback imports for development
    print("Warning: Using fallback imports - some features may be limited")
    
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
    
    # FedAdam hyperparameters (aligned with standard implementation)
    DEFAULT_ETA = 0.1  # Server-side learning rate
    DEFAULT_ETA_L = 0.1  # Client-side learning rate  
    DEFAULT_BETA_1 = 0.9  # Momentum parameter
    DEFAULT_BETA_2 = 0.99  # Second moment parameter
    DEFAULT_TAU = 1e-9  # Controls algorithm's degree of adaptability
    
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

def setup_server_environment():
    """Setup server environment with optimal configurations"""
    
    # Device detection with fallback strategies
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU - consider GPU for better performance")
    
    # Memory optimization
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # For performance
    
    # Random seed management
    try:
        set_seed(42)
    except:
        torch.manual_seed(42)
        np.random.seed(42)
    
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
    """Create fit metrics aggregation function for medical metrics"""
    def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """Aggregate training metrics from clients"""
        if not metrics:
            return {}
        
        # Extract metrics and weights
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        # Weighted aggregation of metrics
        aggregated_metrics = {}
        
        # Standard metrics
        metric_names = ['train_loss', 'train_dice', 'train_iou', 'learning_rate']
        
        for metric_name in metric_names:
            weighted_sum = 0.0
            total_weight = 0
            
            for num_examples, client_metrics in metrics:
                if metric_name in client_metrics:
                    weighted_sum += int(client_metrics[metric_name]) * num_examples
                    total_weight += num_examples
            
            if total_weight > 0:
                aggregated_metrics[f"avg_{metric_name}"] = weighted_sum / total_weight
        
        # Additional aggregated metrics
        aggregated_metrics["total_examples"] = total_examples
        aggregated_metrics["num_clients"] = len(metrics)
        
        return aggregated_metrics
    
    return aggregate_fit_metrics

def create_evaluate_metrics_aggregation_fn() -> Callable:
    """Create evaluate metrics aggregation function for medical metrics"""
    def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """Aggregate evaluation metrics from clients"""
        if not metrics:
            return {}
        
        # Extract metrics and weights
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        # Weighted aggregation of metrics
        aggregated_metrics = {}
        
        # Medical evaluation metrics
        metric_names = [
            'eval_loss', 'eval_dice', 'eval_iou', 'eval_sensitivity', 
            'eval_specificity', 'eval_precision', 'eval_recall'
        ]
        
        for metric_name in metric_names:
            weighted_sum = 0.0
            total_weight = 0
            
            for num_examples, client_metrics in metrics:
                if metric_name in client_metrics:
                    weighted_sum += int(client_metrics[metric_name]) * num_examples
                    total_weight += num_examples
            
            if total_weight > 0:
                aggregated_metrics[f"avg_{metric_name}"] = weighted_sum / total_weight
        
        # Additional aggregated metrics
        aggregated_metrics["total_examples"] = total_examples
        aggregated_metrics["num_clients"] = len(metrics)
        
        return aggregated_metrics
    
    return aggregate_evaluate_metrics

# 4. MODEL AND DATA FUNCTIONS (UNCHANGED)

def create_global_model(config: Dict[str, Any]):
    """Create and initialize global model for server-side evaluation"""
    
    try:
        # Try to create the actual model
        from src.models.mlp_model import RobustMedVFL_UNet
        
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
    """Create evaluation function for server-side model evaluation"""
    
    def evaluate(server_round: int, parameters: Parameters, config_dict: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on test data"""
        
        if test_dataloader is None:
            print(f"No test data available for global evaluation")
            return None
        
        try:
            # Set model parameters
            params_dict = zip(global_model.state_dict().keys(), parameters_to_ndarrays(parameters))
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            global_model.load_state_dict(state_dict, strict=False)
            global_model.eval()
            
            # Evaluation metrics
            total_loss = 0.0
            total_dice = 0.0
            total_iou = 0.0
            num_batches = 0
            
            try:
                from src.models.mlp_model import CombinedLoss
                criterion = CombinedLoss().to(DEVICE)
            except ImportError:
                criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    try:
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            images, masks = batch[0], batch[1]
                        else:
                            continue
                        
                        images = images.to(DEVICE)
                        masks = masks.to(DEVICE)
                        
                        # Forward pass
                        outputs = global_model(images)
                        
                        # Handle different output formats
                        if isinstance(outputs, (list, tuple)):
                            main_output = outputs[0]
                        else:
                            main_output = outputs
                        
                        # Compute loss
                        loss = criterion(main_output, masks)
                        total_loss += loss.item()
                        
                        # Compute Dice score
                        pred_masks = torch.argmax(main_output, dim=1)
                        
                        # Foreground Dice (classes 1, 2, 3)
                        fg_pred = (pred_masks > 0).float()
                        fg_true = (masks > 0).float()
                        
                        intersection = (fg_pred * fg_true).sum()
                        union = fg_pred.sum() + fg_true.sum()
                        
                        if union > 0:
                            dice = (2.0 * intersection) / union
                            total_dice += dice.item()
                        
                        # IoU computation
                        if union > 0:
                            iou = intersection / (union - intersection + 1e-8)
                            total_iou += iou.item()
                        
                        num_batches += 1
                        
                        # Limit evaluation batches for efficiency
                        if batch_idx >= 20:  # Evaluate on first 20 batches
                            break
                            
                    except Exception as e:
                        print(f"Error in evaluation batch {batch_idx}: {e}")
                        continue
            
            if num_batches == 0:
                return None
            
            # Average metrics
            avg_loss = total_loss / num_batches
            avg_dice = total_dice / num_batches
            avg_iou = total_iou / num_batches
            
            eval_metrics = {
                "global_eval_loss": avg_loss,
                "global_eval_dice": avg_dice,
                "global_eval_iou": avg_iou,
                "eval_batches": num_batches,
            }
            
            print(f"Global evaluation - Round {server_round}:")
            print(f" Global Dice (FG): {avg_dice:.4f}")
            print(f" Global Loss: {avg_loss:.4f}")
            
            return float(avg_loss), dict(eval_metrics)
            
        except Exception as e:
            print(f"Error in global evaluation: {e}")
            return None
    
    return evaluate

def load_global_test_data(config: Dict[str, Any]) -> Optional[DataLoader]:
    """Load global test dataset for server-side evaluation"""
    
    try:
        # Try to load test data from configured path
        test_data_path = config.get('test_data_path', '/Users/alvinluong/Documents/Federated_Learning/ACDC/database/testing')
        
        if os.path.exists(test_data_path):
            print(f"Loading global test data from: {test_data_path}")
            
            # Create test dataloader using ACDC dataset
            try:
                from src.data.loader import create_acdc_dataloader
                test_dataloader = create_acdc_dataloader(
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
                            print(f"✓ Global test data loaded: {dataset_size} samples")
                        except TypeError:
                            print(f"✓ Global test data loaded successfully")
                    else:
                        print(f"✓ Global test data loaded successfully")
                except (TypeError, AttributeError):
                    print(f"✓ Global test data loaded successfully")
                
                return test_dataloader
                
            except ImportError:
                print("Warning: create_acdc_dataloader not available - using dummy data")
                return None
        else:
            print(f"Warning: Test data path not found: {test_data_path}")
            return None
            
    except Exception as e:
        print(f"Error loading global test data: {e}")
        return None

# 5. SERVER FACTORY FUNCTION - UPDATED FOR STANDARD FEDADAM

# 6. SERVERAPP CREATION AND EXPORT - UPDATED FOR FLOWER 1.18.0

def server_fn(context: Context) -> fl.server.ServerAppComponents:
    """Create server components for Flower 1.18.0"""
    
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
            'test_data_path': context.run_config.get('test-data-path', None),
            'enable_global_evaluation': context.run_config.get('enable-global-evaluation', True),
            'noise_adaptation': context.run_config.get('noise-adaptation', True),
            'physics_scheduling': context.run_config.get('physics-scheduling', True),
            'progressive_complexity': context.run_config.get('progressive-complexity', True),
            'local_epochs': context.run_config.get('local-epochs', ServerConstants.DEFAULT_LOCAL_EPOCHS),
            'batch_size': context.run_config.get('batch-size', ServerConstants.DEFAULT_BATCH_SIZE),
            'test_batch_size': context.run_config.get('test-batch-size', 4),
            'weight_decay': context.run_config.get('weight-decay', 1e-5),
        }
    except:
        # Fallback to defaults if context is not available
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
            'test_data_path': None,
            'enable_global_evaluation': True,
            'noise_adaptation': True,
            'physics_scheduling': True,
            'progressive_complexity': True,
            'local_epochs': ServerConstants.DEFAULT_LOCAL_EPOCHS,
            'batch_size': ServerConstants.DEFAULT_BATCH_SIZE,
            'test_batch_size': 4,
            'weight_decay': 1e-5,
        }

    print(f"Server configuration loaded:")
    print(f" • Rounds: {server_config['num_rounds']}")
    print(f" • Server LR (eta): {server_config['eta']}")
    print(f" • Client LR (eta_l): {server_config['eta_l']}")

    # Global model setup
    global_model = create_global_model(server_config)

    # Get initial parameters from the global model
    initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in global_model.state_dict().values()])

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

    # Strategy instantiation with standard FedAdam
    strategy = FedAdam(
        # Core FedAdam parameters
        eta=server_config['eta'],  # Server-side learning rate
        eta_l=server_config['eta_l'],  # Client-side learning rate
        beta_1=server_config['beta_1'],  # Momentum parameter
        beta_2=server_config['beta_2'],  # Second moment parameter
        tau=server_config['tau'],  # Controls algorithm's degree of adaptability
        
        # Client selection parameters
        fraction_fit=server_config['fraction_fit'],
        fraction_evaluate=server_config['fraction_evaluate'],
        min_fit_clients=server_config['min_fit_clients'],
        min_evaluate_clients=server_config['min_evaluate_clients'],
        min_available_clients=server_config['min_fit_clients'],  # Use same as min_fit_clients
        
        # Required initial parameters
        initial_parameters=initial_parameters,
        
        # Configuration functions
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
        
        # Evaluation function
        evaluate_fn=evaluate_fn,
        
        # Metrics aggregation functions
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        
        # Error handling
        accept_failures=True,
    )

    print("Standard FedAdam strategy initialized")

    # Server configuration
    config = ServerConfig(
        num_rounds=server_config['num_rounds'],
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
