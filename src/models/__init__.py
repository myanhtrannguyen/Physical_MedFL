"""
Model components for medical image segmentation in federated learning.
"""

from .unet import UNet

__all__ = [
    "UNet", "RobustMedVFL_UNet"
]
