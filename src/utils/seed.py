"""
Seed utilities for reproducibility.
"""

import random
import numpy as np
import torch
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Ensure seed is within valid range (0 to 2^32 - 1)
    seed = int(seed) % (2**32 - 1)
    if seed < 0:
        seed = abs(seed)
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Set random seed to {seed}")


def get_random_state():
    """
    Get current random state for all generators.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state()
    
    return state


def set_random_state(state: dict):
    """
    Set random state for all generators.
    
    Args:
        state: Dictionary containing random states
    """
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy_random'])
    torch.set_rng_state(state['torch_random'])
    
    if torch.cuda.is_available() and 'torch_cuda_random' in state:
        torch.cuda.set_rng_state(state['torch_cuda_random'])
    
    logger.info("Restored random state")


def create_reproducible_generator(seed: Optional[int] = None):
    """
    Create a reproducible random number generator.
    
    Args:
        seed: Seed for the generator
        
    Returns:
        PyTorch generator with fixed seed
    """
    if seed is None:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return generator 