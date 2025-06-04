"""
Source package for Federated Learning Medical Image Segmentation.
"""

__version__ = "0.1.0"

# Main modules
from . import data
from . import models  
from . import fl_core
from . import utils

__all__ = [
    "data",
    "models", 
    "fl_core",
    "utils"
] 