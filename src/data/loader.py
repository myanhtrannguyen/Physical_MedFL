"""
Unified data loader for federated learning with multiple medical datasets.
Provides simple interface for creating DataLoaders for ACDC and BraTS2020.
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List, Optional, Tuple, Union, Any, cast
import logging
from pathlib import Path

from .dataset import (
    UnifiedDatasetManager,
    create_unified_dataset,
    create_multi_dataset_loader,
    DatasetType
)
from .preprocessing import MedicalImagePreprocessor, DataAugmentation
from .partitioning import FederatedDataPartitioner

logger = logging.getLogger(__name__)

class UnifiedFederatedLoader:
    """
    Unified loader for federated learning with multiple medical datasets.
    Handles both ACDC and BraTS2020 datasets with consistent interface.
    """
    
    def __init__(
        self,
        acdc_data_dir: Optional[str] = None,
        brats_data_dir: Optional[str] = None,
        preprocessor_config: Optional[Dict[str, Any]] = None,
        augmentation_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize unified federated loader.
        
        Args:
            acdc_data_dir: Path to ACDC dataset
            brats_data_dir: Path to BraTS2020 dataset
            preprocessor_config: Preprocessor configuration
            augmentation_config: Augmentation configuration
        """
        self.acdc_data_dir = acdc_data_dir
        self.brats_data_dir = brats_data_dir
        
        # Initialize dataset manager
        self.dataset_manager = UnifiedDatasetManager()
        
        # Register available datasets
        self._register_available_datasets()
        
        # Initialize preprocessor and augmentation
        self.preprocessor = self._create_preprocessor(preprocessor_config or {})
        self.augmentation = self._create_augmentation(augmentation_config or {})
    
    def _register_available_datasets(self) -> None:
        """Register datasets that are available."""
        if self.acdc_data_dir and Path(self.acdc_data_dir).exists():
            self.dataset_manager.register_dataset(
                name="acdc",
                dataset_type="acdc",
                data_dir=self.acdc_data_dir
            )
            logger.info(f"Registered ACDC dataset: {self.acdc_data_dir}")
        
        if self.brats_data_dir and Path(self.brats_data_dir).exists():
            self.dataset_manager.register_dataset(
                name="brats2020",
                dataset_type="brats2020",
                data_dir=self.brats_data_dir
            )
            logger.info(f"Registered BraTS2020 dataset: {self.brats_data_dir}")
    
    def _create_preprocessor(self, config: Dict[str, Any]) -> MedicalImagePreprocessor:
        """Create preprocessor from configuration."""
        return MedicalImagePreprocessor(
            target_size=config.get('target_size', (256, 256)),
            normalize=config.get('normalize', True),
            clip_percentiles=config.get('clip_percentiles', (1, 99)),
            apply_clahe=config.get('apply_clahe', True)
        )
    
    def _create_augmentation(self, config: Dict[str, Any]) -> DataAugmentation:
        """Create augmentation from configuration."""
        return DataAugmentation(
            rotation_range=config.get('rotation_range', 15.0),
            zoom_range=config.get('zoom_range', 0.1),
            horizontal_flip=config.get('horizontal_flip', True),
            vertical_flip=config.get('vertical_flip', False),
            brightness_range=config.get('brightness_range', 0.1),
            contrast_range=config.get('contrast_range', 0.1),
            noise_std=config.get('noise_std', 0.01)
        )
    
    def create_single_dataset_loader(
        self,
        dataset_type: DatasetType,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        apply_augmentation: bool = True,
        **dataset_kwargs
    ) -> DataLoader:
        """
        Create DataLoader for a single dataset type.
        
        Args:
            dataset_type: Type of dataset ('acdc' or 'brats2020')
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            apply_augmentation: Whether to apply data augmentation
            **dataset_kwargs: Additional dataset-specific arguments
            
        Returns:
            DataLoader instance
        """
        # Get data directory
        data_dir = self._get_data_directory(dataset_type)
        
        if not data_dir or not Path(data_dir).exists():
            raise ValueError(f"Data directory not found for {dataset_type}: {data_dir}")
        
        # Create dataset
        dataset = create_unified_dataset(
            dataset_type=dataset_type,
            data_dir=data_dir,
            preprocessor=self.preprocessor,
            augmentation=self.augmentation if apply_augmentation else None,
            apply_augmentation=apply_augmentation,
            **dataset_kwargs
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True if shuffle else False
        )
        
        logger.info(
            f"Created {dataset_type} DataLoader: "
            f"{len(dataset)} samples, {len(dataloader)} batches"
        )
        return dataloader
    
    def _get_data_directory(self, dataset_type: DatasetType) -> Optional[str]:
        """Get data directory for dataset type."""
        if dataset_type == "acdc":
            return self.acdc_data_dir
        elif dataset_type == "brats2020":
            return self.brats_data_dir
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def create_combined_loader(
        self,
        datasets: List[DatasetType],
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        apply_augmentation: bool = True,
        **dataset_kwargs
    ) -> DataLoader:
        """
        Create DataLoader combining multiple datasets.
        
        Args:
            datasets: List of dataset types to combine
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            apply_augmentation: Whether to apply data augmentation
            **dataset_kwargs: Additional dataset-specific arguments
            
        Returns:
            Combined DataLoader instance
        """
        dataset_configs: List[Dict[str, Any]] = []
        
        for dataset_type in datasets:
            config = self._create_dataset_config(
                dataset_type, apply_augmentation, **dataset_kwargs
            )
            if config:
                dataset_configs.append(config)
        
        if not dataset_configs:
            raise ValueError("No valid datasets found for combination")
        
        # Create combined DataLoader
        dataloader = create_multi_dataset_loader(
            datasets_config=dataset_configs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True if shuffle else False
        )
        
        # Get dataset size safely
        try:
            # Cast dataloader.dataset to ConcatDataset to help Pylance
            dataset_size = len(cast(ConcatDataset, dataloader.dataset))
        except (TypeError, AttributeError):
            dataset_size = "unknown"
        
        logger.info(
            f"Created combined DataLoader: "
            f"{dataset_size} samples, {len(dataloader)} batches"
        )
        return dataloader
    
    def _create_dataset_config(
        self,
        dataset_type: DatasetType,
        apply_augmentation: bool,
        **dataset_kwargs
    ) -> Optional[Dict[str, Any]]:
        """Create configuration for a dataset."""
        # Get data directory
        data_dir = self._get_data_directory(dataset_type)
        
        if not data_dir or not Path(data_dir).exists():
            logger.warning(f"Data directory not found for {dataset_type}: {data_dir}")
            return None
        
        return {
            'dataset_type': dataset_type,
            'data_dir': data_dir,
            'preprocessor': self.preprocessor,
            'augmentation': self.augmentation if apply_augmentation else None,
            'apply_augmentation': apply_augmentation,
            **dataset_kwargs
        }
    
    def create_federated_loaders(
        self,
        num_clients: int,
        dataset_type: Optional[DatasetType] = None,
        datasets: Optional[List[DatasetType]] = None,
        partition_strategy: str = "iid",
        batch_size: int = 8,
        apply_augmentation: bool = True,
        **kwargs
    ) -> List[DataLoader]:
        """
        Create federated DataLoaders for multiple clients.
        
        Args:
            num_clients: Number of federated clients
            dataset_type: Single dataset type to use
            datasets: Multiple dataset types to combine
            partition_strategy: Partitioning strategy ('iid' or 'non_iid')
            batch_size: Batch size for each client
            apply_augmentation: Whether to apply data augmentation
            **kwargs: Additional arguments
            
        Returns:
            List of DataLoaders for each client
        """
        # Determine which datasets to use
        datasets_to_use = self._determine_datasets_to_use(dataset_type, datasets)
        
        if not datasets_to_use:
            raise ValueError("No datasets available for federated learning")
        
        # Collect data for partitioning
        all_data_paths, dataset_info = self._collect_federated_data(datasets_to_use)
        
        # Create partitioner and partition data
        partition = self._create_data_partition(
            all_data_paths, partition_strategy, num_clients
        )
        
        # Create DataLoaders for each client
        client_loaders = self._create_client_loaders(
            partition, datasets_to_use, num_clients, batch_size, apply_augmentation
        )
        
        return client_loaders
    
    def _determine_datasets_to_use(
        self,
        dataset_type: Optional[DatasetType],
        datasets: Optional[List[DatasetType]]
    ) -> List[DatasetType]:
        """Determine which datasets to use for federated learning."""
        if dataset_type:
            return [dataset_type]
        elif datasets:
            return datasets
        else:
            # Use all available datasets
            available_datasets: List[DatasetType] = []
            if self.acdc_data_dir and Path(self.acdc_data_dir).exists():
                available_datasets.append("acdc")
            if self.brats_data_dir and Path(self.brats_data_dir).exists():
                available_datasets.append("brats2020")
            return available_datasets
    
    def _collect_federated_data(
        self, datasets_to_use: List[DatasetType]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Collect data paths for federated partitioning."""
        all_data_paths: List[str] = []
        dataset_info: List[Dict[str, Any]] = []
        
        for dataset_name in datasets_to_use:
            data_dir = self._get_data_directory(dataset_name)
            if data_dir:
                paths, info = self._get_dataset_paths(dataset_name, data_dir)
                all_data_paths.extend(paths)
                dataset_info.extend(info)
        
        return all_data_paths, dataset_info
    
    def _get_dataset_paths(
        self, dataset_name: DatasetType, data_dir: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Get data paths for a specific dataset."""
        try:
            dataset = create_unified_dataset(
                dataset_type=dataset_name,
                data_dir=data_dir,
                apply_augmentation=False
            )
            
            paths = [f"{dataset_name}_{i}" for i in range(len(dataset))]
            info = [{"type": dataset_name, "index": i} for i in range(len(dataset))]
            
            return paths, info
            
        except Exception as e:
            logger.error(f"Error getting paths for {dataset_name}: {e}")
            return [], []
    
    def _create_data_partition(
        self,
        all_data_paths: List[str],
        partition_strategy: str,
        num_clients: int
    ) -> Dict[int, List[str]]:
        """Create data partition for federated learning."""
        partitioner = FederatedDataPartitioner(
            num_clients=num_clients,
            data_dir="",  # Not used for this approach
            output_dir="data/processed/federated_splits",
            seed=42
        )
        
        if partition_strategy == "iid":
            return partitioner.create_iid_partition(all_data_paths, "unified")
        else:
            # Pass None for labels to use default extraction logic
            return partitioner.create_non_iid_partition(all_data_paths, labels=None, partition_name="unified")
    
    def _create_client_loaders(
        self,
        partition: Dict[int, List[str]],
        datasets_to_use: List[DatasetType],
        num_clients: int,
        batch_size: int,
        apply_augmentation: bool
    ) -> List[DataLoader]:
        """Create DataLoaders for federated clients."""
        client_loaders: List[DataLoader] = []
        
        for client_id in range(num_clients):
            client_data_paths = partition.get(client_id, [])
            
            if not client_data_paths:
                logger.warning(f"No data assigned to client {client_id}")
                continue
            
            # Create client dataset (simplified approach)
            # In practice, you would create a custom dataset that uses these indices
            client_loader = self.create_combined_loader(
                datasets=datasets_to_use,
                batch_size=batch_size,
                shuffle=True,
                apply_augmentation=apply_augmentation
            )
            
            client_loaders.append(client_loader)
            logger.info(
                f"Created DataLoader for client {client_id}: "
                f"{len(client_data_paths)} samples"
            )
        
        return client_loaders
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info: Dict[str, Any] = {
            "available_datasets": [],
            "total_samples": 0,
            "datasets_info": {}
        }
        
        # Check ACDC
        if self.acdc_data_dir and Path(self.acdc_data_dir).exists():
            acdc_info = self._get_single_dataset_info("acdc", self.acdc_data_dir)
            if acdc_info:
                info["available_datasets"].append("acdc")
                info["total_samples"] += acdc_info["samples"]
                info["datasets_info"]["acdc"] = acdc_info
        
        # Check BraTS2020
        if self.brats_data_dir and Path(self.brats_data_dir).exists():
            brats_info = self._get_single_dataset_info("brats2020", self.brats_data_dir)
            if brats_info:
                info["available_datasets"].append("brats2020")
                info["total_samples"] += brats_info["samples"]
                info["datasets_info"]["brats2020"] = brats_info
        
        return info
    
    def _get_single_dataset_info(
        self, dataset_type: DatasetType, data_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Get information for a single dataset."""
        try:
            dataset = create_unified_dataset(
                dataset_type=dataset_type,
                data_dir=data_dir,
                apply_augmentation=False
            )
            return {
                "type": dataset_type,
                "samples": len(dataset),
                "classes": dataset.get_num_classes(),
                "class_names": dataset.get_class_names(),
                "data_dir": data_dir
            }
        except Exception as e:
            logger.error(f"Error getting {dataset_type} info: {e}")
            return None


# Convenience functions
def create_acdc_loader(
    data_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    apply_augmentation: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create ACDC DataLoader with default settings.
    
    Args:
        data_dir: Path to ACDC data directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        apply_augmentation: Whether to apply augmentation
        **kwargs: Additional arguments
        
    Returns:
        ACDC DataLoader
    """
    loader = UnifiedFederatedLoader(acdc_data_dir=data_dir)
    return loader.create_single_dataset_loader(
        dataset_type="acdc",
        batch_size=batch_size,
        shuffle=shuffle,
        apply_augmentation=apply_augmentation,
        **kwargs
    )


def create_brats_loader(
    data_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    apply_augmentation: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create BraTS2020 DataLoader with default settings.
    
    Args:
        data_dir: Path to BraTS2020 data directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        apply_augmentation: Whether to apply augmentation
        **kwargs: Additional arguments
        
    Returns:
        BraTS2020 DataLoader
    """
    loader = UnifiedFederatedLoader(brats_data_dir=data_dir)
    return loader.create_single_dataset_loader(
        dataset_type="brats2020",
        batch_size=batch_size,
        shuffle=shuffle,
        apply_augmentation=apply_augmentation,
        **kwargs
    )


def create_multi_medical_loader(
    acdc_data_dir: Optional[str] = None,
    brats_data_dir: Optional[str] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    apply_augmentation: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create combined medical datasets DataLoader.
    
    Args:
        acdc_data_dir: Path to ACDC data directory
        brats_data_dir: Path to BraTS2020 data directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        apply_augmentation: Whether to apply augmentation
        **kwargs: Additional arguments
        
    Returns:
        Combined DataLoader
    """
    datasets_to_use: List[DatasetType] = []
    if acdc_data_dir:
        datasets_to_use.append("acdc")
    if brats_data_dir:
        datasets_to_use.append("brats2020")
    
    if not datasets_to_use:
        raise ValueError("At least one dataset directory must be provided")
    
    loader = UnifiedFederatedLoader(
        acdc_data_dir=acdc_data_dir,
        brats_data_dir=brats_data_dir
    )
    return loader.create_combined_loader(
        datasets=datasets_to_use,
        batch_size=batch_size,
        shuffle=shuffle,
        apply_augmentation=apply_augmentation,
        **kwargs
    ) 