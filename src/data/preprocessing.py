"""
Data preprocessing utilities for medical image segmentation.
Handles cleaning, normalization, and augmentation of medical images.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Tuple, Optional, Union
import cv2
from skimage import exposure, filters
import logging

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """Preprocessor for medical images with normalization and augmentation."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        clip_percentiles: Tuple[float, float] = (1, 99),
        apply_clahe: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize images to [0, 1]
            clip_percentiles: Percentiles for intensity clipping
            apply_clahe: Whether to apply CLAHE for contrast enhancement
        """
        self.target_size = target_size
        self.normalize = normalize
        self.clip_percentiles = clip_percentiles
        self.apply_clahe = apply_clahe
        
        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image
        """
        # Ensure image is 2D
        if len(image.shape) > 2:
            image = image.squeeze()
        
        # Handle different data types
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        elif image.dtype == np.uint16:
            image = image.astype(np.float32)
        
        # Intensity clipping for outlier removal
        if self.clip_percentiles:
            p_low, p_high = np.percentile(image, self.clip_percentiles)
            image = np.clip(image, p_low, p_high)
        
        # Normalize to [0, 1]
        if self.normalize:
            image_min, image_max = image.min(), image.max()
            if image_max > image_min:
                image = (image - image_min) / (image_max - image_min)
            else:
                image = np.zeros_like(image)
        
        # Apply CLAHE for contrast enhancement
        if self.apply_clahe and image.max() <= 1.0:
            # Convert to uint8 for CLAHE
            image_uint8 = (image * 255).astype(np.uint8)
            image_uint8 = self.clahe.apply(image_uint8)
            image = image_uint8.astype(np.float32) / 255.0
        
        # Resize image
        if image.shape != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def preprocess_mask(self, mask: np.ndarray, num_classes: int = 4) -> np.ndarray:
        """
        Preprocess a segmentation mask.
        
        Args:
            mask: Input mask array
            num_classes: Number of segmentation classes
            
        Returns:
            Preprocessed mask
        """
        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # Resize mask using nearest neighbor interpolation
        if mask.shape != self.target_size:
            mask = cv2.resize(
                mask.astype(np.float32), 
                self.target_size, 
                interpolation=cv2.INTER_NEAREST
            )
        
        # Ensure mask values are within valid range
        mask = np.clip(mask, 0, num_classes - 1)
        
        return mask.astype(np.int64)


class DataAugmentation:
    """Data augmentation for medical images and masks."""
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        zoom_range: float = 0.1,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1,
        noise_std: float = 0.01
    ):
        """
        Initialize augmentation parameters.
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            zoom_range: Maximum zoom factor
            horizontal_flip: Whether to apply horizontal flipping
            vertical_flip: Whether to apply vertical flipping
            brightness_range: Brightness adjustment range
            contrast_range: Contrast adjustment range
            noise_std: Standard deviation for Gaussian noise
        """
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
    
    def augment_pair(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to image-mask pair.
        
        Args:
            image: Input image
            mask: Input mask
            
        Returns:
            Augmented image and mask
        """
        # Convert to tensors for easier manipulation
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image_tensor = self._rotate_tensor(image_tensor, angle)
            mask_tensor = self._rotate_tensor(mask_tensor, angle, is_mask=True)
        
        # Random zoom
        if self.zoom_range > 0:
            zoom_factor = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            image_tensor = self._zoom_tensor(image_tensor, zoom_factor)
            mask_tensor = self._zoom_tensor(mask_tensor, zoom_factor, is_mask=True)
        
        # Random flips
        if self.horizontal_flip and np.random.random() > 0.5:
            image_tensor = torch.flip(image_tensor, dims=[3])
            mask_tensor = torch.flip(mask_tensor, dims=[3])
        
        if self.vertical_flip and np.random.random() > 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            mask_tensor = torch.flip(mask_tensor, dims=[2])
        
        # Convert back to numpy
        image = image_tensor.squeeze().numpy()
        mask = mask_tensor.squeeze().numpy()
        
        # Apply intensity augmentations only to image
        if self.brightness_range > 0:
            brightness_factor = np.random.uniform(
                1 - self.brightness_range, 1 + self.brightness_range
            )
            image = np.clip(image * brightness_factor, 0, 1)
        
        if self.contrast_range > 0:
            contrast_factor = np.random.uniform(
                1 - self.contrast_range, 1 + self.contrast_range
            )
            image = np.clip((image - 0.5) * contrast_factor + 0.5, 0, 1)
        
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image, mask.astype(np.int64)
    
    def _rotate_tensor(
        self, 
        tensor: torch.Tensor, 
        angle: float, 
        is_mask: bool = False
    ) -> torch.Tensor:
        """Rotate tensor by given angle."""
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Create rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Create affine grid
        grid = F.affine_grid(
            rotation_matrix, 
            list(tensor.size()), 
            align_corners=False
        )
        
        # Apply rotation
        mode = 'nearest' if is_mask else 'bilinear'
        rotated = F.grid_sample(
            tensor, 
            grid, 
            mode=mode, 
            align_corners=False,
            padding_mode='zeros'
        )
        
        return rotated
    
    def _zoom_tensor(
        self, 
        tensor: torch.Tensor, 
        zoom_factor: float, 
        is_mask: bool = False
    ) -> torch.Tensor:
        """Zoom tensor by given factor."""
        # Create zoom matrix
        zoom_matrix = torch.tensor([
            [zoom_factor, 0, 0],
            [0, zoom_factor, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Create affine grid
        grid = F.affine_grid(
            zoom_matrix, 
            list(tensor.size()), 
            align_corners=False
        )
        
        # Apply zoom
        mode = 'nearest' if is_mask else 'bilinear'
        zoomed = F.grid_sample(
            tensor, 
            grid, 
            mode=mode, 
            align_corners=False,
            padding_mode='zeros'
        )
        
        return zoomed


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images to zero mean and unit variance.
    
    Args:
        batch: Batch of images [B, C, H, W]
        
    Returns:
        Normalized batch
    """
    batch_size = batch.size(0)
    batch_flat = batch.view(batch_size, -1)
    
    mean = batch_flat.mean(dim=1, keepdim=True)
    std = batch_flat.std(dim=1, keepdim=True)
    
    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)
    
    normalized = (batch_flat - mean) / std
    return normalized.view_as(batch)


def compute_intensity_statistics(images: np.ndarray) -> dict:
    """
    Compute intensity statistics for a collection of images.
    
    Args:
        images: Array of images [N, H, W] or [N, H, W, C]
        
    Returns:
        Dictionary with intensity statistics
    """
    if len(images.shape) == 4:
        images = images.squeeze(-1)
    
    stats = {
        'mean': np.mean(images),
        'std': np.std(images),
        'min': np.min(images),
        'max': np.max(images),
        'median': np.median(images),
        'percentile_1': np.percentile(images, 1),
        'percentile_99': np.percentile(images, 99)
    }
    
    return stats 