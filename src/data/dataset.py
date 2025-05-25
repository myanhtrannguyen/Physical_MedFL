"""
Custom Dataset classes for medical image segmentation.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, Callable, List
import nibabel as nib
import h5py
import os
from pathlib import Path
import logging

from .preprocessing import MedicalImagePreprocessor, DataAugmentation

logger = logging.getLogger(__name__)


class MedicalSegmentationDataset(Dataset):
    """Dataset for medical image segmentation."""
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        preprocessor: Optional[MedicalImagePreprocessor] = None,
        augmentation: Optional[DataAugmentation] = None,
        num_classes: int = 4,
        apply_augmentation: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            mask_paths: List of paths to mask files (optional)
            preprocessor: Image preprocessor
            augmentation: Data augmentation
            num_classes: Number of segmentation classes
            apply_augmentation: Whether to apply augmentation
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes
        self.apply_augmentation = apply_augmentation
        
        # Initialize preprocessor
        if preprocessor is None:
            self.preprocessor = MedicalImagePreprocessor()
        else:
            self.preprocessor = preprocessor
        
        # Initialize augmentation
        if augmentation is None and apply_augmentation:
            self.augmentation = DataAugmentation()
        else:
            self.augmentation = augmentation
        
        # Validate paths
        self._validate_paths()
        
        logger.info(f"Initialized dataset with {len(self.image_paths)} samples")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, mask) tensors
        """
        # Load image
        image = self._load_image(self.image_paths[idx])
        
        # Load mask if available
        if self.mask_paths is not None:
            mask = self._load_mask(self.mask_paths[idx])
        else:
            # Create dummy mask
            mask = np.zeros(image.shape, dtype=np.int64)
        
        # Preprocess
        image = self.preprocessor.preprocess_image(image)
        mask = self.preprocessor.preprocess_mask(mask, self.num_classes)
        
        # Apply augmentation
        if self.apply_augmentation and self.augmentation is not None:
            image, mask = self.augmentation.augment_pair(image, mask)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor
    
    def _validate_paths(self):
        """Validate that all paths exist."""
        for path in self.image_paths:
            if not os.path.exists(path):
                logger.warning(f"Image path does not exist: {path}")
        
        if self.mask_paths is not None:
            if len(self.mask_paths) != len(self.image_paths):
                raise ValueError("Number of mask paths must match number of image paths")
            
            for path in self.mask_paths:
                if not os.path.exists(path):
                    logger.warning(f"Mask path does not exist: {path}")
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image from file."""
        try:
            if path.endswith('.nii') or path.endswith('.nii.gz'):
                # Load NIfTI file
                nii_img = nib.load(path)  # type: ignore
                image = nii_img.get_fdata()  # type: ignore
            elif path.endswith('.h5') or path.endswith('.hdf5'):
                # Load HDF5 file
                with h5py.File(path, 'r') as f:
                    # Try common dataset names
                    for key in ['image', 'data', 'volume']:
                        if key in f:
                            image = f[key][:]  # type: ignore
                            break
                    else:
                        # Use first dataset
                        key = list(f.keys())[0]
                        image = f[key][:]  # type: ignore
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            return np.array(image).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            # Return dummy image
            return np.zeros((256, 256), dtype=np.float32)
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask from file."""
        try:
            if path.endswith('.nii') or path.endswith('.nii.gz'):
                # Load NIfTI file
                nii_img = nib.load(path)  # type: ignore
                mask = nii_img.get_fdata()  # type: ignore
            elif path.endswith('.h5') or path.endswith('.hdf5'):
                # Load HDF5 file
                with h5py.File(path, 'r') as f:
                    # Try common dataset names
                    for key in ['mask', 'label', 'segmentation', 'gt']:
                        if key in f:
                            mask = f[key][:]  # type: ignore
                            break
                    else:
                        # Use first dataset
                        key = list(f.keys())[0]
                        mask = f[key][:]  # type: ignore
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            return np.array(mask).astype(np.int64)
            
        except Exception as e:
            logger.error(f"Error loading mask {path}: {e}")
            # Return dummy mask
            return np.zeros((256, 256), dtype=np.int64)


class ACDCDataset(MedicalSegmentationDataset):
    """Specialized dataset for ACDC cardiac segmentation."""
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: Optional[List[str]] = None,
        frames: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize ACDC dataset.
        
        Args:
            data_dir: Directory containing ACDC data
            patient_ids: List of patient IDs to include
            frames: List of frame types to include (e.g., ['frame01', 'frame12'])
            **kwargs: Additional arguments for parent class
        """
        self.data_dir = Path(data_dir)
        
        # Collect image and mask paths
        image_paths, mask_paths = self._collect_acdc_paths(patient_ids, frames)
        
        super().__init__(
            image_paths=image_paths,
            mask_paths=mask_paths,
            num_classes=4,  # ACDC has 4 classes: background, RV, myocardium, LV
            **kwargs
        )
    
    def _collect_acdc_paths(
        self, 
        patient_ids: Optional[List[str]] = None,
        frames: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """Collect ACDC image and mask paths."""
        image_paths = []
        mask_paths = []
        
        # Default frames if not specified
        if frames is None:
            frames = ['frame01', 'frame12']  # ED and ES frames
        
        # Find all patient directories
        patient_dirs = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith('patient'):
                if patient_ids is None or item.name in patient_ids:
                    patient_dirs.append(item)
        
        # Collect paths for each patient and frame
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            
            for frame in frames:
                # Look for image and mask files
                image_pattern = f"{patient_id}_{frame}.nii"
                mask_pattern = f"{patient_id}_{frame}_gt.nii"
                
                image_path = patient_dir / image_pattern
                mask_path = patient_dir / mask_pattern
                
                if image_path.exists():
                    image_paths.append(str(image_path))
                    
                    if mask_path.exists():
                        mask_paths.append(str(mask_path))
                    else:
                        logger.warning(f"Mask not found: {mask_path}")
                        mask_paths.append(None)  # Will be handled in dataset
        
        # Filter out None masks
        valid_pairs = [(img, mask) for img, mask in zip(image_paths, mask_paths) if mask is not None]
        if valid_pairs:
            image_paths, mask_paths = zip(*valid_pairs)
            image_paths, mask_paths = list(image_paths), list(mask_paths)
        
        logger.info(f"Found {len(image_paths)} ACDC samples")
        return image_paths, mask_paths


class InMemoryDataset(Dataset):
    """Dataset that loads all data into memory for faster access."""
    
    def __init__(
        self,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
        augmentation: Optional[DataAugmentation] = None,
        apply_augmentation: bool = True
    ):
        """
        Initialize in-memory dataset.
        
        Args:
            images: Array of images [N, H, W]
            masks: Array of masks [N, H, W] (optional)
            augmentation: Data augmentation
            apply_augmentation: Whether to apply augmentation
        """
        self.images = images
        self.masks = masks
        self.apply_augmentation = apply_augmentation
        
        # Initialize augmentation
        if augmentation is None and apply_augmentation:
            self.augmentation = DataAugmentation()
        else:
            self.augmentation = augmentation
        
        logger.info(f"Initialized in-memory dataset with {len(self.images)} samples")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        image = self.images[idx].copy()
        
        if self.masks is not None:
            mask = self.masks[idx].copy()
        else:
            mask = np.zeros(image.shape, dtype=np.int64)
        
        # Apply augmentation
        if self.apply_augmentation and self.augmentation is not None:
            image, mask = self.augmentation.augment_pair(image, mask)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor


def create_dataset_from_paths(
    image_paths: List[str],
    mask_paths: Optional[List[str]] = None,
    dataset_type: str = "file",
    **kwargs
) -> Dataset:
    """
    Factory function to create datasets.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of mask file paths
        dataset_type: Type of dataset ('file', 'acdc', 'memory')
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    if dataset_type == "file":
        return MedicalSegmentationDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            **kwargs
        )
    elif dataset_type == "acdc":
        # For ACDC, image_paths[0] should be the data directory
        return ACDCDataset(
            data_dir=image_paths[0],
            **kwargs
        )
    elif dataset_type == "memory":
        # Load all images into memory
        preprocessor = kwargs.get('preprocessor', MedicalImagePreprocessor())
        
        images = []
        masks = []
        
        for i, img_path in enumerate(image_paths):
            # Load and preprocess image
            if img_path.endswith('.nii') or img_path.endswith('.nii.gz'):
                nii_img = nib.load(img_path)  # type: ignore
                image = np.array(nii_img.get_fdata()).astype(np.float32)  # type: ignore
            else:
                continue
            
            image = preprocessor.preprocess_image(image)
            images.append(image)
            
            # Load mask if available
            if mask_paths and i < len(mask_paths) and mask_paths[i]:
                nii_mask = nib.load(mask_paths[i])  # type: ignore
                mask = np.array(nii_mask.get_fdata()).astype(np.int64)  # type: ignore
                mask = preprocessor.preprocess_mask(mask)
                masks.append(mask)
        
        images = np.array(images)
        masks = np.array(masks) if masks else None
        
        return InMemoryDataset(
            images=images,
            masks=masks,
            **{k: v for k, v in kwargs.items() if k != 'preprocessor'}
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def compute_class_weights(masks: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        masks: Array of masks [N, H, W]
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    # Flatten all masks
    flat_masks = masks.flatten()
    
    # Count pixels for each class
    class_counts = np.bincount(flat_masks, minlength=num_classes)
    
    # Compute weights (inverse frequency)
    total_pixels = len(flat_masks)
    class_weights = total_pixels / (num_classes * class_counts + 1e-8)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return torch.from_numpy(class_weights).float() 