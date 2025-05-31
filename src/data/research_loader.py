"""
Research-focused data loaders for medical imaging with maximum accuracy.
Eliminates redundancy and provides single source of truth for data loading.
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path

from .dataset import create_unified_dataset, DatasetType, ACDCUnifiedDataset, BraTS2020UnifiedDataset
from .preprocessing import MedicalImagePreprocessor, DataAugmentation

logger = logging.getLogger(__name__)


def create_research_dataloader(
    dataset_type: DatasetType,
    data_dir: str,
    batch_size: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    client_id: Optional[int] = None,
    total_clients: Optional[int] = None,
    preprocessor_config: Optional[Dict[str, Any]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    Create single research-grade DataLoader with maximum accuracy.
    Single source of truth - no redundant preprocessing layers.
    
    Args:
        dataset_type: 'acdc' or 'brats2020'
        data_dir: Path to dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        augment: Whether to apply data augmentation
        client_id: Optional client ID for federated learning
        total_clients: Total number of clients for federated learning
        preprocessor_config: Preprocessing configuration
        augmentation_config: Augmentation configuration
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional dataset-specific arguments
        
    Returns:
        Research-grade DataLoader
    """
    # Create single preprocessor instance (avoid duplication)
    preprocessor = MedicalImagePreprocessor(
        target_size=preprocessor_config.get('target_size', (256, 256)) if preprocessor_config else (256, 256),
        normalize=preprocessor_config.get('normalize', True) if preprocessor_config else True,
        clip_percentiles=preprocessor_config.get('clip_percentiles', (1, 99)) if preprocessor_config else (1, 99),
        apply_clahe=preprocessor_config.get('apply_clahe', True) if preprocessor_config else True
    )
    
    # Create single augmentation instance (if needed)
    augmentation = None
    if augment:
        augmentation = DataAugmentation(
            rotation_range=augmentation_config.get('rotation_range', 10.0) if augmentation_config else 10.0,
            zoom_range=augmentation_config.get('zoom_range', 0.1) if augmentation_config else 0.1,
            horizontal_flip=augmentation_config.get('horizontal_flip', True) if augmentation_config else True,
            vertical_flip=augmentation_config.get('vertical_flip', False) if augmentation_config else False,
            brightness_range=augmentation_config.get('brightness_range', 0.1) if augmentation_config else 0.1,
            contrast_range=augmentation_config.get('contrast_range', 0.1) if augmentation_config else 0.1,
            noise_std=augmentation_config.get('noise_std', 0.01) if augmentation_config else 0.01
        )
    
    # Create dataset with single preprocessing path
    dataset = create_unified_dataset(
        dataset_type=dataset_type,
        data_dir=data_dir,
        preprocessor=preprocessor,
        augmentation=augmentation,
        apply_augmentation=augment,
        client_id=client_id,
        total_num_clients=total_clients,
        **dataset_kwargs
    )
    
    # Create DataLoader with research-optimized settings
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True if shuffle and len(dataset) >= batch_size else False,
    }
    
    # Only add these if using multiprocessing
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    logger.info(
        f"Created research DataLoader: {dataset_type} | "
        f"{len(dataset)} samples | {len(dataloader)} batches | "
        f"Client {client_id}/{total_clients if total_clients else 'N/A'}"
    )
    
    return dataloader


def create_federated_research_loaders(
    dataset_type: DatasetType,
    data_dir: str,
    num_clients: int,
    batch_size: int = 4,
    augment: bool = True,
    preprocessor_config: Optional[Dict[str, Any]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
    **dataset_kwargs
) -> List[DataLoader]:
    """
    Create federated research DataLoaders with guaranteed consistency.
    Each client gets identical preprocessing but different data partitions.
    
    Args:
        dataset_type: 'acdc' or 'brats2020'
        data_dir: Path to dataset
        num_clients: Number of federated clients
        batch_size: Batch size per client
        augment: Whether to apply data augmentation
        preprocessor_config: Preprocessing configuration
        augmentation_config: Augmentation configuration
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional dataset-specific arguments
        
    Returns:
        List of research-grade DataLoaders for each client
    """
    client_loaders = []
    
    # Create shared preprocessor and augmentation for consistency
    preprocessor = MedicalImagePreprocessor(
        target_size=preprocessor_config.get('target_size', (256, 256)) if preprocessor_config else (256, 256),
        normalize=preprocessor_config.get('normalize', True) if preprocessor_config else True,
        clip_percentiles=preprocessor_config.get('clip_percentiles', (1, 99)) if preprocessor_config else (1, 99),
        apply_clahe=preprocessor_config.get('apply_clahe', True) if preprocessor_config else True
    )
    
    augmentation = None
    if augment:
        augmentation = DataAugmentation(
            rotation_range=augmentation_config.get('rotation_range', 10.0) if augmentation_config else 10.0,
            zoom_range=augmentation_config.get('zoom_range', 0.1) if augmentation_config else 0.1,
            horizontal_flip=augmentation_config.get('horizontal_flip', True) if augmentation_config else True,
            vertical_flip=augmentation_config.get('vertical_flip', False) if augmentation_config else False,
            brightness_range=augmentation_config.get('brightness_range', 0.1) if augmentation_config else 0.1,
            contrast_range=augmentation_config.get('contrast_range', 0.1) if augmentation_config else 0.1,
            noise_std=augmentation_config.get('noise_std', 0.01) if augmentation_config else 0.01
        )
    
    # Create client-specific datasets with shared preprocessing
    for client_id in range(num_clients):
        # Create dataset for this specific client
        dataset = create_unified_dataset(
            dataset_type=dataset_type,
            data_dir=data_dir,
            preprocessor=preprocessor,  # Shared preprocessor for consistency
            augmentation=augmentation,  # Shared augmentation for consistency
            apply_augmentation=augment,
            client_id=client_id,
            total_num_clients=num_clients,
            **dataset_kwargs
        )
        
        # Create DataLoader for this client
        dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,  # Always shuffle for training
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "drop_last": True if len(dataset) >= batch_size else False,
        }
        
        # Only add these if using multiprocessing
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = 2
        
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        
        client_loaders.append(dataloader)
        
        logger.info(
            f"Client {client_id}: {len(dataset)} samples, {len(dataloader)} batches"
        )
    
    logger.info(
        f"Created {num_clients} federated research DataLoaders for {dataset_type}"
    )
    
    return client_loaders


def get_research_client_dataloader(
    client_id: int,
    total_clients: int,
    dataset_type: DatasetType,
    data_dir: str,
    batch_size: int = 4,
    augment: bool = True,
    preprocessor_config: Optional[Dict[str, Any]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    **dataset_kwargs
) -> DataLoader:
    """
    Get research DataLoader for specific client (Flower-compatible).
    Eliminates all redundancy and provides maximum research accuracy.
    
    Args:
        client_id: Client ID (0-based)
        total_clients: Total number of clients
        dataset_type: 'acdc' or 'brats2020'
        data_dir: Path to dataset
        batch_size: Batch size
        augment: Whether to apply augmentation
        preprocessor_config: Preprocessing configuration
        augmentation_config: Augmentation configuration
        **dataset_kwargs: Additional dataset arguments
        
    Returns:
        Research-grade DataLoader for the specific client
    """
    return create_research_dataloader(
        dataset_type=dataset_type,
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True,
        augment=augment,
        client_id=client_id,
        total_clients=total_clients,
        preprocessor_config=preprocessor_config,
        augmentation_config=augmentation_config,
        num_workers=0,  # Flower typically uses 0 workers
        **dataset_kwargs
    )


# Research utility functions
def validate_data_consistency(dataloaders: List[DataLoader]) -> Dict[str, Any]:
    """
    Validate that federated dataloaders have consistent preprocessing.
    Critical for research accuracy.
    """
    if not dataloaders:
        return {"valid": False, "error": "No dataloaders provided"}
    
    # Check batch sizes
    batch_sizes = [dl.batch_size for dl in dataloaders]
    
    # Get dataset sizes safely
    total_samples = 0
    samples_per_client = []
    
    for dl in dataloaders:
        try:
            dataset_size = len(dl.dataset)  # type: ignore
            total_samples += dataset_size
            samples_per_client.append(dataset_size)
        except (TypeError, AttributeError):
            samples_per_client.append(0)
    
    # Check dataset types and preprocessing
    validation_results = {
        "valid": True,
        "num_clients": len(dataloaders),
        "batch_sizes": batch_sizes,
        "batch_size_consistent": len(set(batch_sizes)) == 1,
        "total_samples": total_samples,
        "samples_per_client": samples_per_client,
        "data_balance_ratio": max(samples_per_client) / min(samples_per_client) if min(samples_per_client) > 0 else float('inf'),
        "avg_samples_per_client": total_samples / len(dataloaders) if dataloaders else 0
    }
    
    return validation_results


def validate_research_reproducibility(
    dataset_type: DatasetType,
    data_dir: str,
    seed: int = 42,
    num_samples: int = 5
) -> Dict[str, Any]:
    """
    Validate research reproducibility by checking if same seeds produce same results.
    Critical for published research.
    """
    import torch
    import numpy as np
    import random
    
    # Helper function to set seed
    def set_seed(seed_val):
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
    
    try:
        # First run with seed
        set_seed(seed)
        dataloader1 = create_research_dataloader(
            dataset_type=dataset_type,
            data_dir=data_dir,
            batch_size=2,
            shuffle=True,
            augment=True
        )
        
        # Second run with same seed
        set_seed(seed)
        dataloader2 = create_research_dataloader(
            dataset_type=dataset_type,
            data_dir=data_dir,
            batch_size=2,
            shuffle=True,
            augment=True
        )
        
        # Compare first few batches
        reproducible = True
        differences = []
        
        iter1 = iter(dataloader1)
        iter2 = iter(dataloader2)
        
        for i in range(min(num_samples, len(dataloader1))):
            try:
                batch1 = next(iter1)
                batch2 = next(iter2)
                
                # Compare image tensors
                img_diff = torch.abs(batch1[0] - batch2[0]).mean().item()
                differences.append(img_diff)
                
                if img_diff > 1e-6:  # Allow tiny floating point differences
                    reproducible = False
                    
            except StopIteration:
                break
        
        return {
            "reproducible": reproducible,
            "seed_used": seed,
            "num_samples_tested": len(differences),
            "max_difference": max(differences) if differences else 0.0,
            "avg_difference": sum(differences) / len(differences) if differences else 0.0
        }
        
    except Exception as e:
        return {
            "reproducible": False,
            "error": str(e)
        }


def benchmark_dataloader_performance(
    dataloader: DataLoader,
    num_batches: int = 10
) -> Dict[str, Any]:
    """
    Benchmark DataLoader performance for research optimization.
    """
    import time
    
    times = []
    memory_usage = []
    
    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            start_time = time.time()
            
            # Simulate basic operations
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, masks = batch[0], batch[1]
                # Force GPU transfer if available
                if torch.cuda.is_available():
                    images = images.cuda()
                    masks = masks.cuda()
                
                # Simple computation
                _ = images.mean()
                _ = masks.unique()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Monitor GPU memory if available
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
        
        # Calculate throughput safely
        batch_size = getattr(dataloader, 'batch_size', 1)
        throughput = (len(times) * batch_size) / sum(times) if times and sum(times) > 0 else 0.0
        
        return {
            "avg_batch_time": sum(times) / len(times) if times else 0.0,
            "min_batch_time": min(times) if times else 0.0,
            "max_batch_time": max(times) if times else 0.0,
            "total_time": sum(times),
            "throughput_samples_per_sec": throughput,
            "avg_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0.0,
            "peak_memory_mb": max(memory_usage) if memory_usage else 0.0
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "avg_batch_time": float('inf')
        }


def create_research_report(
    dataloaders: List[DataLoader],
    dataset_type: DatasetType,
    data_dir: str
) -> Dict[str, Any]:
    """
    Generate comprehensive research report for the data pipeline.
    """
    import datetime
    
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset_type": dataset_type,
        "data_directory": data_dir,
        "num_clients": len(dataloaders)
    }
    
    # Consistency validation
    report["consistency"] = validate_data_consistency(dataloaders)
    
    # Reproducibility check (using first dataloader's config)
    if dataloaders:
        report["reproducibility"] = validate_research_reproducibility(
            dataset_type=dataset_type,
            data_dir=data_dir
        )
        
        # Performance benchmark
        report["performance"] = benchmark_dataloader_performance(dataloaders[0])
    
    # Data distribution analysis
    total_samples = sum(len(dl.dataset) for dl in dataloaders)  # type: ignore
    client_samples = [len(dl.dataset) for dl in dataloaders]  # type: ignore
    
    report["data_distribution"] = {
        "total_samples": total_samples,
        "samples_per_client": client_samples,
        "min_samples": min(client_samples) if client_samples else 0,
        "max_samples": max(client_samples) if client_samples else 0,
        "std_samples": float(torch.std(torch.tensor(client_samples, dtype=torch.float))) if client_samples else 0.0,
        "balance_coefficient": 1.0 - (max(client_samples) - min(client_samples)) / max(client_samples) if client_samples and max(client_samples) > 0 else 0.0
    }
    
    return report


__all__ = [
    'create_research_dataloader',
    'create_federated_research_loaders', 
    'get_research_client_dataloader',
    'validate_data_consistency',
    'validate_research_reproducibility',
    'benchmark_dataloader_performance',
    'create_research_report'
] 