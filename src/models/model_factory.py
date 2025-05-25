"""
Model factory for creating different neural network architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_model(
    model_type: str,
    model_config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model to create ('unet', 'kan', 'mlp')
        model_config: Configuration dictionary for the model
        device: Device to place the model on
        
    Returns:
        Model instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_type = model_type.lower()
    
    if model_type == 'unet' or model_type == 'robustmedvfl_unet':
        model = create_unet_model(model_config)
    elif model_type == 'kan':
        model = create_kan_model(model_config)
    elif model_type == 'mlp':
        model = create_mlp_model(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    logger.info(f"Created {model_type} model with {count_parameters(model)} parameters")
    
    return model


def create_unet_model(config: Dict[str, Any]) -> nn.Module:
    """Create U-Net model."""
    try:
        from .mlp_model import OptimizedRobustMedVFL_UNet
        
        model = OptimizedRobustMedVFL_UNet(
            n_channels=config.get('n_channels', 1),
            n_classes=config.get('n_classes', 4),
            dropout_rate=config.get('dropout_rate', 0.1)
        )
        
        return model
        
    except ImportError as e:
        logger.error(f"Failed to import U-Net model: {e}")
        raise


def create_kan_model(config: Dict[str, Any]) -> nn.Module:
    """Create KAN model."""
    try:
        # Try to import KAN model - this might fail if not available
        from .kan_model import KANModel  # type: ignore
        
        # Extract KAN-specific parameters
        kan_config = {
            'input_size': config.get('input_size', 256 * 256),
            'hidden_sizes': config.get('hidden_sizes', [128, 64]),
            'output_size': config.get('output_size', 4),
            'grid_size': config.get('grid_size', 5),
            'spline_order': config.get('spline_order', 3)
        }
        
        model = KANModel(**kan_config)  # type: ignore
        return model
        
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import KAN model: {e}")
        # Create a simple MLP as fallback
        logger.warning("Using MLP as fallback for KAN model")
        return create_mlp_model(config)


def create_mlp_model(config: Dict[str, Any]) -> nn.Module:
    """Create simple MLP model."""
    input_size = config.get('input_size', 256 * 256)
    hidden_sizes = config.get('hidden_sizes', [512, 256, 128])
    output_size = config.get('output_size', 4)
    dropout_rate = config.get('dropout_rate', 0.1)
    
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_size, hidden_sizes[0]))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    
    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    
    # Output layer
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
    model = nn.Sequential(*layers)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get information about a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': str(model.__class__.__name__)
    }
    
    return info


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Load model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    logger.info(f"Loaded model checkpoint from {checkpoint_path}")
    
    return model


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None
):
    """Save model checkpoint."""
    checkpoint: Dict[str, Any] = {
        'model_state_dict': model.state_dict(),
        'model_info': get_model_info(model)
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved model checkpoint to {checkpoint_path}")


# Default model configurations
DEFAULT_CONFIGS = {
    'unet': {
        'n_channels': 1,
        'n_classes': 4,
        'dropout_rate': 0.1
    },
    'kan': {
        'input_size': 256 * 256,
        'hidden_sizes': [128, 64],
        'output_size': 4,
        'grid_size': 5,
        'spline_order': 3
    },
    'mlp': {
        'input_size': 256 * 256,
        'hidden_sizes': [512, 256, 128],
        'output_size': 4,
        'dropout_rate': 0.1
    }
}


def get_default_config(model_type: str) -> Dict[str, Any]:
    """Get default configuration for a model type."""
    model_type = model_type.lower()
    if model_type in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[model_type].copy()
    else:
        raise ValueError(f"No default config for model type: {model_type}")


def create_model_from_config_file(config_path: str, device: Optional[torch.device] = None) -> nn.Module:
    """Create model from configuration file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'unet')
    
    return create_model(model_type, model_config, device) 