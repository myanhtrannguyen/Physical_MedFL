"""
Unified dataset system for multiple medical imaging datasets.
Supports ACDC (cardiac segmentation) and BraTS2020 (brain tumor segmentation).
"""

from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import numpy as np
import h5py
import nibabel as nib
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image
from typing import Tuple, Optional, List, Dict, Union, Literal, Any, cast
from pathlib import Path
import logging
import json

from .preprocessing import MedicalImagePreprocessor, DataAugmentation

logger = logging.getLogger(__name__)

DatasetType = Literal["acdc", "brats2020"]

class BaseUnifiedDataset(Dataset, ABC):
    """Base class for unified medical imaging datasets."""
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: Optional[MedicalImagePreprocessor] = None,
        augmentation: Optional[DataAugmentation] = None,
        apply_augmentation: bool = True,
        cache_data: bool = False
    ) -> None:
        """
        Initialize base dataset.
        
        Args:
            data_dir: Directory containing the dataset
            preprocessor: Image preprocessor
            augmentation: Data augmentation
            apply_augmentation: Whether to apply augmentation
            cache_data: Whether to cache loaded data in memory
        """
        self.data_dir = Path(data_dir)
        self.apply_augmentation = apply_augmentation
        self.cache_data = cache_data
        
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
        
        # Data cache
        self._cache: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = {} if cache_data else None
        
        # Initialize dataset-specific data
        self.data_paths: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self._initialize_dataset()
    
    @abstractmethod
    def _initialize_dataset(self) -> None:
        """Initialize dataset-specific data paths and metadata."""
        pass
    
    @abstractmethod
    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single sample (image, mask) from dataset."""
        pass
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """Get number of segmentation classes."""
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Get class names for segmentation."""
        pass
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, mask) tensors
        """
        # Check cache first
        if self._cache is not None and idx in self._cache:
            image, mask = self._cache[idx]
        else:
            # Load sample
            image, mask = self._load_sample(idx)
            
            # Preprocess
            image = self.preprocessor.preprocess_image(image)
            mask = self.preprocessor.preprocess_mask(mask, self.get_num_classes())
            
            # Cache if enabled
            if self._cache is not None:
                self._cache[idx] = (image.copy(), mask.copy())
        
        # Apply augmentation
        if self.apply_augmentation and self.augmentation is not None:
            image, mask = self.augmentation.augment_pair(image, mask)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata information for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample metadata
        """
        if idx < len(self.data_paths):
            return {
                "index": idx,
                "path": str(self.data_paths[idx]),
                "dataset_type": self.__class__.__name__,
                "num_classes": self.get_num_classes(),
                "class_names": self.get_class_names()
            }
        return {}


class ACDCUnifiedDataset(BaseUnifiedDataset):
    """ACDC cardiac segmentation dataset."""
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: Optional[List[str]] = None,
        frames: Optional[List[str]] = None,
        client_id: Optional[int] = None,
        total_num_clients: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize ACDC dataset.
        
        Args:
            data_dir: Directory containing ACDC data
            patient_ids: List of patient IDs to include (overrides client_id logic if provided)
            frames: List of frame types to include (e.g., ['frame01', 'frame12'])
            client_id: Optional ID of the client for data partitioning.
            total_num_clients: Optional total number of clients for data partitioning.
            **kwargs: Additional arguments for parent class
        """
        self.patient_ids = patient_ids
        self.frames = frames or ['frame01', 'frame12']  # ED and ES frames
        self.client_id = client_id
        self.total_num_clients = total_num_clients
        super().__init__(data_dir, **kwargs)
    
    def _initialize_dataset(self) -> None:
        """Initialize ACDC dataset paths."""
        if not self.data_dir.exists():
            raise ValueError(f"ACDC data directory not found: {self.data_dir}")
        
        # Collect patient directories based on client_id or specified patient_ids
        patient_dirs = self._get_patient_directories()
        
        # Collect image and mask paths
        for patient_dir in patient_dirs:
            self._process_patient_directory(patient_dir)
        
        logger.info(f"ACDC dataset initialized for client {self.client_id if self.client_id is not None else 'all'}: {len(self.data_paths)} samples")
    
    def _get_patient_directories(self) -> List[Path]:
        """Get list of patient directories to process, considering client_id."""
        all_patient_dirs = sorted([d for d in self.data_dir.iterdir() 
                                   if d.is_dir() and d.name.startswith('patient')])
        
        if self.patient_ids: # Explicit patient_ids override client_id logic
            logger.info(f"Using specified patient IDs for ACDC: {self.patient_ids}")
            return [d for d in all_patient_dirs if d.name in self.patient_ids]
        
        if self.client_id is not None and self.total_num_clients is not None and self.total_num_clients > 0:
            num_all_patients = len(all_patient_dirs)
            if num_all_patients == 0:
                logger.warning("No patient directories found in ACDC data_dir.")
                return []

            # Ensure client_id is valid
            if not (0 <= self.client_id < self.total_num_clients):
                logger.error(f"Invalid client_id {self.client_id} for {self.total_num_clients} clients.")
                return []

            samples_per_client = num_all_patients // self.total_num_clients
            remainder = num_all_patients % self.total_num_clients
            
            start_idx = self.client_id * samples_per_client + min(self.client_id, remainder)
            num_samples_for_this_client = samples_per_client + (1 if self.client_id < remainder else 0)
            end_idx = start_idx + num_samples_for_this_client
            
            client_specific_dirs = all_patient_dirs[start_idx:end_idx]
            logger.info(f"ACDC Client {self.client_id}/{self.total_num_clients}: assigned {len(client_specific_dirs)} patient dirs (range {start_idx}-{end_idx-1} from {num_all_patients} total).")
            return client_specific_dirs
        
        # Default: use all patient directories if no client partitioning or specific IDs
        logger.info("ACDC: No specific patient_ids or client partitioning, using all patient directories.")
        return all_patient_dirs
    
    def _process_patient_directory(self, patient_dir: Path) -> None:
        """Process a single patient directory to find image/mask pairs."""
        for frame in self.frames:
            # Look for image and mask files
            image_files = list(patient_dir.glob(f"*{frame}.nii*"))
            mask_files = list(patient_dir.glob(f"*{frame}_gt.nii*"))
            
            for img_file in image_files:
                mask_file = self._find_corresponding_mask(mask_files, frame)
                
                if mask_file and mask_file.exists():
                    self.data_paths.append({
                        'image': str(img_file),
                        'mask': str(mask_file),
                        'patient_id': patient_dir.name,
                        'frame': frame
                    })
                else:
                    logger.warning(f"Mask not found for {img_file}")
    
    def _find_corresponding_mask(self, mask_files: List[Path], frame: str) -> Optional[Path]:
        """Find corresponding mask file for a given frame."""
        for mask in mask_files:
            if frame in mask.name and 'gt' in mask.name:
                return mask
        return None
    
    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load ACDC sample."""
        sample_info = self.data_paths[idx]
        
        try:
            # Load image
            img_nii = cast(Nifti1Image, nib_load(sample_info['image']))
            image = img_nii.get_fdata().astype(np.float32)
            
            # Load mask
            mask_nii = cast(Nifti1Image, nib_load(sample_info['mask']))
            mask = mask_nii.get_fdata().astype(np.int64)
            
            # Take middle slice if 3D
            if len(image.shape) > 2:
                slice_idx = image.shape[2] // 2
                image = image[:, :, slice_idx]
                mask = mask[:, :, slice_idx]
            
            return image, mask
            
        except Exception as e:
            logger.error(f"Error loading ACDC sample {idx}: {e}")
            # Return dummy data
            return np.zeros((256, 256), dtype=np.float32), np.zeros((256, 256), dtype=np.int64)
    
    def get_num_classes(self) -> int:
        """Get number of classes in ACDC dataset."""
        return 4  # Background, RV, Myocardium, LV
    
    def get_class_names(self) -> List[str]:
        """Get class names for ACDC dataset."""
        return ['Background', 'RV', 'Myocardium', 'LV']


class BraTS2020UnifiedDataset(BaseUnifiedDataset):
    """BraTS2020 brain tumor segmentation dataset."""
    
    def __init__(
        self,
        data_dir: str,
        volume_ids: Optional[List[int]] = None,
        slice_range: Optional[Tuple[int, int]] = None,
        modalities: Optional[List[str]] = None,
        client_id: Optional[int] = None,
        total_num_clients: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize BraTS2020 dataset.
        
        Args:
            data_dir: Directory containing BraTS2020 H5 files
            volume_ids: List of volume IDs to include (overrides client_id logic if provided)
            slice_range: Range of slices to include (min, max)
            modalities: List of modalities to use (default: all 4)
            client_id: Optional ID of the client for data partitioning.
            total_num_clients: Optional total number of clients for data partitioning.
            **kwargs: Additional arguments for parent class
        """
        self.specified_volume_ids = volume_ids
        self.slice_range = slice_range
        self.modalities = modalities or ['flair', 't1', 't1ce', 't2']
        self.client_id = client_id
        self.total_num_clients = total_num_clients
        super().__init__(data_dir, **kwargs)
    
    def _initialize_dataset(self) -> None:
        """Initialize BraTS2020 dataset paths."""
        if not self.data_dir.exists():
            raise ValueError(f"BraTS2020 data directory not found: {self.data_dir}")
        
        all_h5_files = sorted(list(self.data_dir.glob("*.h5"))) # Sort for consistent partitioning

        files_to_process = []

        if self.specified_volume_ids: # Explicit volume_ids override client_id logic
            logger.info(f"BraTS: Using specified volume_ids: {self.specified_volume_ids}")
            # This requires parsing file names to get volume_id for filtering
            temp_files_for_specified_ids = []
            for h5_file in all_h5_files:
                try:
                    parts = h5_file.stem.split('_')
                    if len(parts) >= 4 and parts[0] == 'volume':
                        volume_id = int(parts[1])
                        if volume_id in self.specified_volume_ids:
                            temp_files_for_specified_ids.append(h5_file)
                except (ValueError, IndexError):
                    logger.warning(f"BraTS: Could not parse volume_id from {h5_file.name} when using specified_volume_ids.")
            files_to_process = temp_files_for_specified_ids
        elif self.client_id is not None and self.total_num_clients is not None and self.total_num_clients > 0:
            num_all_files = len(all_h5_files)
            if num_all_files == 0:
                logger.warning("No H5 files found in BraTS data_dir.")
                self.data_paths = []
                return

            if not (0 <= self.client_id < self.total_num_clients):
                logger.error(f"Invalid client_id {self.client_id} for BraTS with {self.total_num_clients} clients.")
                self.data_paths = []
                return
            
            samples_per_client = num_all_files // self.total_num_clients
            remainder = num_all_files % self.total_num_clients
            
            start_idx = self.client_id * samples_per_client + min(self.client_id, remainder)
            num_samples_for_this_client = samples_per_client + (1 if self.client_id < remainder else 0)
            end_idx = start_idx + num_samples_for_this_client
            
            files_to_process = all_h5_files[start_idx:end_idx]
            logger.info(f"BraTS Client {self.client_id}/{self.total_num_clients}: assigned {len(files_to_process)} H5 files (range {start_idx}-{end_idx-1} from {num_all_files} total).")
        else:
            logger.info("BraTS: No specific volume_ids or client partitioning, using all H5 files.")
            files_to_process = all_h5_files
        
        for h5_file in files_to_process:
            # _process_h5_file now implicitly uses self.slice_range if set
            self._process_h5_file(h5_file)
        
        # Sort by volume and slice ID (if data_paths were populated)
        if self.data_paths:
            self.data_paths.sort(key=lambda x: (x.get('volume_id', -1), x.get('slice_id', -1)))
        
        logger.info(f"BraTS2020 dataset initialized for client {self.client_id if self.client_id is not None else 'all'}: {len(self.data_paths)} samples")
    
    def _process_h5_file(self, h5_file: Path) -> None:
        """Process a single H5 file, considering self.slice_range."""
        try:
            parts = h5_file.stem.split('_')
            if len(parts) >= 4 and parts[0] == 'volume' and parts[2] == 'slice':
                volume_id = int(parts[1])
                slice_id = int(parts[3])
                
                # Apply slice_range filter if it's specified
                if self.slice_range:
                    min_slice, max_slice = self.slice_range
                    if not (min_slice <= slice_id <= max_slice):
                        return # Skip this slice if it's outside the range
                
                # Add to data paths if it passes all filters
                self.data_paths.append({
                    'file': str(h5_file),
                    'volume_id': volume_id,
                    'slice_id': slice_id
                })
                    
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse filename {h5_file.name}: {e}")
    
    def _should_include_file(self, volume_id: int, slice_id: int) -> bool:
        """
        DEPRECATED: This method's logic is now incorporated into _initialize_dataset and _process_h5_file.
        """
        # Filter by volume IDs (if specified_volume_ids is being used AND no client partitioning)
        if self.specified_volume_ids and self.client_id is None:
            if volume_id not in self.specified_volume_ids:
                return False
        
        # Filter by slice range
        if self.slice_range:
            min_slice, max_slice = self.slice_range
            if not (min_slice <= slice_id <= max_slice):
                return False
        
        return True
    
    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load BraTS2020 sample."""
        sample_info = self.data_paths[idx]
        
        try:
            with h5py.File(sample_info['file'], 'r') as f:
                # Load image (4 modalities: FLAIR, T1, T1ce, T2)
                image_dataset = f['image']
                image_data = np.array(image_dataset)  # Shape: (240, 240, 4)
                
                # Load mask (3 classes: enhancing tumor, tumor core, whole tumor)
                mask_dataset = f['mask']
                mask_data = np.array(mask_dataset)  # Shape: (240, 240, 3)
                
                # Convert multi-channel mask to single-channel with class labels
                mask = self._convert_mask_to_labels(mask_data)
                
                # Process image modalities
                image = self._process_image_modalities(image_data)
                
                return image, mask
                
        except Exception as e:
            logger.error(f"Error loading BraTS2020 sample {idx}: {e}")
            # Return dummy data
            return np.zeros((240, 240), dtype=np.float32), np.zeros((240, 240), dtype=np.int64)
    
    def _convert_mask_to_labels(self, mask_data: np.ndarray) -> np.ndarray:
        """Convert multi-channel mask to single-channel with class labels."""
        mask = np.zeros(mask_data.shape[:2], dtype=np.int64)
        for i in range(mask_data.shape[2]):
            mask[mask_data[:, :, i] > 0] = i + 1
        return mask
    
    def _process_image_modalities(self, image_data: np.ndarray) -> np.ndarray:
        """Process image modalities based on configuration."""
        if len(self.modalities) == 4:
            # Use all modalities - take mean for single channel
            return np.mean(image_data, axis=2).astype(np.float32)
        else:
            # Select specific modalities (implementation can be extended)
            return image_data[:, :, 0].astype(np.float32)  # Use first modality
    
    def get_num_classes(self) -> int:
        """Get number of classes in BraTS2020 dataset."""
        return 4  # Background + 3 tumor classes
    
    def get_class_names(self) -> List[str]:
        """Get class names for BraTS2020 dataset."""
        return ['Background', 'Enhancing Tumor', 'Tumor Core', 'Whole Tumor']


# Convenience functions
def create_unified_dataset(
    dataset_type: DatasetType,
    data_dir: str,
    preprocessor: Optional[MedicalImagePreprocessor] = None,
    augmentation: Optional[DataAugmentation] = None,
    client_id: Optional[int] = None,
    total_num_clients: Optional[int] = None,
    **kwargs
) -> BaseUnifiedDataset:
    """
    Create a unified dataset instance.
    
    Args:
        dataset_type: Type of dataset ('acdc' or 'brats2020')
        data_dir: Path to dataset directory
        preprocessor: Image preprocessor
        augmentation: Data augmentation
        client_id: Optional client ID for data partitioning.
        total_num_clients: Optional total number of clients for partitioning.
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Dataset instance
    """
    # Add client_id and total_num_clients to dataset_specific_kwargs
    dataset_specific_kwargs = kwargs.copy()
    if client_id is not None:
        dataset_specific_kwargs['client_id'] = client_id
    if total_num_clients is not None:
        dataset_specific_kwargs['total_num_clients'] = total_num_clients

    if dataset_type == 'acdc':
        return ACDCUnifiedDataset(
            data_dir=data_dir,
            preprocessor=preprocessor,
            augmentation=augmentation,
            **dataset_specific_kwargs 
        )
    elif dataset_type == 'brats2020':
        return BraTS2020UnifiedDataset(
            data_dir=data_dir,
            preprocessor=preprocessor,
            augmentation=augmentation,
            **dataset_specific_kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}") 