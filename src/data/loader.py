"""
DataLoader utilities for medical image segmentation.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Tuple, List
import logging

from .dataset import (
    MedicalSegmentationDataset, 
    ACDCDataset, 
    InMemoryDataset,
    create_dataset_from_paths,
    compute_class_weights
)
from .preprocessing import MedicalImagePreprocessor, DataAugmentation

logger = logging.getLogger(__name__)


def create_dataloader(
    images: np.ndarray,
    masks: Optional[np.ndarray] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader from numpy arrays.
    
    Args:
        images: Array of images [N, H, W]
        masks: Array of masks [N, H, W] (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader instance
    """
    # Create augmentation if needed
    augmentation = DataAugmentation() if augment else None
    
    # Create dataset
    dataset = InMemoryDataset(
        images=images,
        masks=masks,
        augmentation=augmentation,
        apply_augmentation=augment
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
    return dataloader


def create_dataloader_from_paths(
    image_paths: List[str],
    mask_paths: Optional[List[str]] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    dataset_type: str = "file",
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader from file paths.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of mask file paths
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        dataset_type: Type of dataset to create
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader instance
    """
    # Create augmentation if needed
    augmentation = DataAugmentation() if augment else None
    
    # Create dataset
    dataset = create_dataset_from_paths(
        image_paths=image_paths,
        mask_paths=mask_paths,
        dataset_type=dataset_type,
        augmentation=augmentation,
        apply_augmentation=augment,
        **dataset_kwargs
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")  # type: ignore
    return dataloader


def create_acdc_dataloader(
    data_dir: str,
    patient_ids: Optional[List[str]] = None,
    frames: Optional[List[str]] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for ACDC dataset.
    
    Args:
        data_dir: Directory containing ACDC data
        patient_ids: List of patient IDs to include
        frames: List of frame types to include
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader instance
    """
    # Create augmentation if needed
    augmentation = DataAugmentation() if augment else None
    
    # Create ACDC dataset
    dataset = ACDCDataset(
        data_dir=data_dir,
        patient_ids=patient_ids,
        frames=frames,
        augmentation=augmentation,
        apply_augmentation=augment,
        **dataset_kwargs
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"Created ACDC DataLoader with {len(dataset)} samples, batch_size={batch_size}")
    return dataloader


def create_federated_dataloaders(
    client_data_paths: dict,
    mask_data_paths: Optional[dict] = None,
    batch_size: int = 8,
    augment_train: bool = True,
    augment_test: bool = False,
    num_workers: int = 0,
    train_test_split: float = 0.8,
    **dataset_kwargs
) -> Tuple[dict, dict]:
    """
    Create federated DataLoaders for multiple clients.
    
    Args:
        client_data_paths: Dict mapping client_id to list of image paths
        mask_data_paths: Dict mapping client_id to list of mask paths
        batch_size: Batch size
        augment_train: Whether to augment training data
        augment_test: Whether to augment test data
        num_workers: Number of worker processes
        train_test_split: Fraction of data for training
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_dataloaders, test_dataloaders) dicts
    """
    train_dataloaders = {}
    test_dataloaders = {}
    
    for client_id, image_paths in client_data_paths.items():
        # Get mask paths for this client
        mask_paths = mask_data_paths.get(client_id) if mask_data_paths else None
        
        # Split data into train/test
        n_train = int(len(image_paths) * train_test_split)
        
        train_image_paths = image_paths[:n_train]
        test_image_paths = image_paths[n_train:]
        
        if mask_paths:
            train_mask_paths = mask_paths[:n_train]
            test_mask_paths = mask_paths[n_train:]
        else:
            train_mask_paths = None
            test_mask_paths = None
        
        # Create training DataLoader
        if len(train_image_paths) > 0:
            train_dataloaders[client_id] = create_dataloader_from_paths(
                image_paths=train_image_paths,
                mask_paths=train_mask_paths,
                batch_size=batch_size,
                shuffle=True,
                augment=augment_train,
                num_workers=num_workers,
                **dataset_kwargs
            )
        
        # Create test DataLoader
        if len(test_image_paths) > 0:
            test_dataloaders[client_id] = create_dataloader_from_paths(
                image_paths=test_image_paths,
                mask_paths=test_mask_paths,
                batch_size=batch_size,
                shuffle=False,
                augment=augment_test,
                num_workers=num_workers,
                **dataset_kwargs
            )
        
        logger.info(f"Client {client_id}: {len(train_image_paths)} train, {len(test_image_paths)} test samples")
    
    return train_dataloaders, test_dataloaders


def compute_dataset_statistics(dataloader: DataLoader) -> dict:
    """
    Compute statistics for a dataset.
    
    Args:
        dataloader: DataLoader to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    all_images = []
    all_masks = []
    
    for batch_images, batch_masks in dataloader:
        all_images.append(batch_images.numpy())
        all_masks.append(batch_masks.numpy())
    
    # Concatenate all batches
    all_images = np.concatenate(all_images, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Compute image statistics
    image_stats = {
        'mean': np.mean(all_images),
        'std': np.std(all_images),
        'min': np.min(all_images),
        'max': np.max(all_images),
        'shape': all_images.shape
    }
    
    # Compute mask statistics
    unique_labels, label_counts = np.unique(all_masks, return_counts=True)
    mask_stats = {
        'unique_labels': unique_labels.tolist(),
        'label_counts': label_counts.tolist(),
        'label_distribution': {
            int(label): int(count) for label, count in zip(unique_labels, label_counts)
        },
        'shape': all_masks.shape
    }
    
    stats = {
        'num_samples': len(all_images),
        'image_stats': image_stats,
        'mask_stats': mask_stats
    }
    
    logger.info(f"Dataset statistics: {stats}")
    return stats


def create_weighted_sampler(
    dataset,  # type: ignore
    class_weights: Optional[torch.Tensor] = None
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a weighted sampler for imbalanced datasets.
    
    Args:
        dataset: Dataset to sample from
        class_weights: Pre-computed class weights
        
    Returns:
        WeightedRandomSampler instance
    """
    if class_weights is None:
        # Compute class weights from dataset
        all_masks = []
        for i in range(len(dataset)):
            _, mask = dataset[i]
            all_masks.append(mask.numpy())
        
        all_masks = np.array(all_masks)
        num_classes = len(np.unique(all_masks))
        class_weights = compute_class_weights(all_masks, num_classes)
    
    # Compute sample weights
    sample_weights = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        # Use the weight of the most frequent class in the mask
        unique_labels, counts = np.unique(mask.numpy(), return_counts=True)
        dominant_class = unique_labels[np.argmax(counts)]
        sample_weights.append(class_weights[dominant_class].item())
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    logger.info(f"Created weighted sampler with {len(sample_weights)} samples")
    return sampler


def collate_fn(batch):
    """
    Custom collate function for handling variable-sized data.
    
    Args:
        batch: List of (image, mask) tuples
        
    Returns:
        Batched tensors
    """
    images, masks = zip(*batch)
    
    # Stack tensors
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    return images, masks 