import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import List, Dict, Optional, Any, Tuple, cast
import json
import logging
from pathlib import Path

# Assuming these are correctly importable from your project structure
# Adjust these imports based on your actual project structure
from .dataset import create_unified_dataset, DatasetType
# BaseUnifiedDataset is not directly used by this function but by create_unified_dataset
from .preprocessing import MedicalImagePreprocessor, DataAugmentation

logger = logging.getLogger(__name__)

def get_client_dataloader_direct(
    client_id: int,
    total_num_clients: int, 
    dataset_types_for_client: List[DatasetType],
    base_data_dirs: Dict[DatasetType, str],
    batch_size: int,
    apply_augmentation: bool,
    preprocessor: MedicalImagePreprocessor,
    augmentation: Optional[DataAugmentation],
    num_workers: int = 0,
    # Pass dataset-specific kwargs if their __init__ methods need them beyond client_id/total_num_clients
    acdc_kwargs: Optional[Dict[str, Any]] = None,
    brats_kwargs: Optional[Dict[str, Any]] = None
) -> DataLoader:
    """
    Creates a DataLoader for a specific client by directly instantiating 
    dataset classes that handle client-specific data loading internally.
    """
    client_datasets = []
    if acdc_kwargs is None:
        acdc_kwargs = {}
    if brats_kwargs is None:
        brats_kwargs = {}

    for ds_type in dataset_types_for_client:
        current_data_dir = base_data_dirs.get(ds_type)
        if not current_data_dir or not Path(current_data_dir).exists():
            logger.warning(f"Data directory for {ds_type} not found or not configured: {current_data_dir}. Skipping for client {client_id}.")
            continue

        ds_specific_kwargs = {}
        if ds_type == "acdc":
            ds_specific_kwargs = acdc_kwargs
        elif ds_type == "brats2020":
            ds_specific_kwargs = brats_kwargs
        
        # The updated create_unified_dataset now handles client_id and total_num_clients
        dataset_instance = create_unified_dataset(
            dataset_type=ds_type,
            data_dir=current_data_dir,
            preprocessor=preprocessor,
            augmentation=augmentation if apply_augmentation else None,
            apply_augmentation=apply_augmentation,
            client_id=client_id,
            total_num_clients=total_num_clients,
            **ds_specific_kwargs
        )

        if len(dataset_instance) > 0:
            client_datasets.append(dataset_instance)
            logger.info(f"Client {client_id}: loaded {len(dataset_instance)} samples for {ds_type}.")
        else:
            logger.warning(f"Client {client_id} got an empty dataset for {ds_type} (dir: {current_data_dir}).")

    if not client_datasets:
        logger.warning(f"Client {client_id} has no data to load after attempting direct instantiation for types: {dataset_types_for_client}.")
        # Return an empty DataLoader
        # Create a dummy empty ConcatDataset with a dummy empty Subset to satisfy DataLoader constructor
        empty_concat = ConcatDataset([])
        empty_subset = Subset(empty_concat, [])
        return DataLoader(empty_subset, batch_size=batch_size, num_workers=num_workers)

    # If only one dataset type was successfully loaded, no need for ConcatDataset
    if len(client_datasets) == 1:
        final_client_dataset = client_datasets[0]
    else:
        final_client_dataset = ConcatDataset(client_datasets)
    
    logger.info(f"Client {client_id} direct dataloader: total {len(final_client_dataset)} samples.")
    
    return DataLoader(
        final_client_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if len(final_client_dataset) >= batch_size else False
    ) 