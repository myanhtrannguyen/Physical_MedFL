"""
Optimized medical dataset for federated learning.
Enhanced error handling and performance optimization.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from nibabel.loadsave import load as nib_load  # Explicit import from correct module
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable
import cv2
import logging
from functools import lru_cache
import os

logger = logging.getLogger(__name__)


class SimpleMedicalDataset(Dataset):
    """
    Optimized medical dataset with robust error handling and caching.
    Enhanced for federated learning with better missing file handling.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        cache_size: int = 100,
        augment_fn: Optional[Callable] = None,
        client_id: Optional[int] = None,
        num_clients: Optional[int] = None
    ):
        """
        Args:
            data_dir: Path to ACDC dataset directory
            image_size: Target image size (H, W)
            cache_size: Number of samples to cache in memory
            augment_fn: Optional augmentation function
            client_id: Client ID for federated partitioning
            num_clients: Total number of clients
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment_fn = augment_fn
        
        # Collect all valid samples with improved validation
        self.samples = self._collect_samples()
        
        # Apply federated partitioning if specified
        if client_id is not None and num_clients is not None:
            self.samples = self._partition_data(client_id, num_clients)
        
        # Setup LRU cache for loaded samples
        self._load_sample_cached = lru_cache(maxsize=cache_size)(self._load_sample_raw)
        
        logger.info(f"Dataset initialized: {len(self.samples)} samples, client_id: {client_id}")
    
    def _collect_samples(self) -> List[Dict[str, str]]:
        """Collect all valid ACDC samples with enhanced error handling."""
        samples = []
        missing_files = []
        
        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return samples
        
        # Get patient directories
        patient_dirs = sorted([d for d in self.data_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('patient')])
        
        if not patient_dirs:
            logger.warning(f"No patient directories found in {self.data_dir}")
            return samples
        
        for patient_dir in patient_dirs:
            # Parse Info.cfg to get correct ED/ES frame numbers
            info_cfg_path = patient_dir / 'Info.cfg'
            ed_frame, es_frame = self._parse_info_cfg(info_cfg_path)
            
            if ed_frame == -1 or es_frame == -1:
                logger.warning(f"Could not parse Info.cfg for {patient_dir.name}, trying all frames")
                # Fallback: try common frame numbers
                frames_to_try = [(1, 'ED'), (8, 'ES')]  # Common ACDC frame numbers
            else:
                frames_to_try = [(ed_frame, 'ED'), (es_frame, 'ES')]
            
            # Process frames
            for frame_num, frame_type in frames_to_try:
                image_found, mask_found = False, False
                image_path, mask_path = "", ""
                
                # Try different file extensions and formats
                for ext in ['.nii.gz', '.nii']:
                    if not image_found:
                        potential_image = patient_dir / f"{patient_dir.name}_frame{frame_num:02d}{ext}"
                        if self._validate_file(potential_image):
                            image_path = str(potential_image)
                            image_found = True
                    
                    if not mask_found:
                        potential_mask = patient_dir / f"{patient_dir.name}_frame{frame_num:02d}_gt{ext}"
                        if self._validate_file(potential_mask):
                            mask_path = str(potential_mask)
                            mask_found = True
                
                # Only add if both files exist and are valid
                if image_found and mask_found:
                    samples.append({
                        'image_path': image_path,
                        'mask_path': mask_path,
                        'patient_id': patient_dir.name,
                        'frame': f"frame{frame_num:02d}",
                        'cardiac_phase': frame_type
                    })
                else:
                    missing_key = f"{patient_dir.name} frame {frame_num:02d}"
                    if missing_key not in missing_files:
                        missing_files.append(missing_key)
        
        # Report missing files as a single summary to avoid spam
        if missing_files:
            # Group by patient to count
            missing_by_patient = {}
            for missing in missing_files:
                patient = missing.split()[0]
                if patient not in missing_by_patient:
                    missing_by_patient[patient] = []
                missing_by_patient[patient].append(missing)
            
            # Single summary warning instead of individual warnings
            total_missing_patients = len(missing_by_patient)
            total_missing_files = len(missing_files)
            logger.warning(f"Missing data for {total_missing_patients} patients ({total_missing_files} files total)")
        
        logger.info(f"Collected {len(samples)} valid samples from {len(patient_dirs)} patients")
        return samples
    
    def _validate_file(self, file_path: Path) -> bool:
        """Validate if file exists and is a valid NIfTI file."""
        if not file_path.exists():
            return False
        
        # Quick size check - NIfTI files should be larger than 1KB
        try:
            if file_path.stat().st_size < 1024:
                return False
        except OSError:
            return False
        
        # Try to load header only for validation (faster than full load)
        try:
            nib_load(str(file_path))
            return True
        except Exception:
            return False
    
    def _parse_info_cfg(self, info_cfg_path: Path) -> Tuple[int, int]:
        """Parse Info.cfg to get ED and ES frame numbers with better error handling."""
        ed_frame, es_frame = -1, -1
        
        if not info_cfg_path.exists():
            return ed_frame, es_frame
        
        try:
            # Try multiple parsing methods
            with open(info_cfg_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith('ED:') or line.startswith('ED ='):
                    ed_frame = int(''.join(filter(str.isdigit, line)))
                elif line.startswith('ES:') or line.startswith('ES ='):
                    es_frame = int(''.join(filter(str.isdigit, line)))
            
            # Fallback: try configparser approach
            if ed_frame == -1 or es_frame == -1:
                import configparser
                parser = configparser.ConfigParser()
                
                with open(info_cfg_path, 'r') as f:
                    config_content = f.read()
                    if not config_content.strip().startswith('['):
                        config_content = '[DEFAULT]\n' + config_content
                
                parser.read_string(config_content)
                if 'ED' in parser['DEFAULT']:
                    ed_frame = int(parser['DEFAULT']['ED'])
                if 'ES' in parser['DEFAULT']:
                    es_frame = int(parser['DEFAULT']['ES'])
            
        except Exception as e:
            logger.debug(f"Could not parse {info_cfg_path}: {e}")
        
        return ed_frame, es_frame
    
    def _partition_data(self, client_id: int, num_clients: int) -> List[Dict[str, str]]:
        """Improved IID partitioning with better distribution."""
        if num_clients <= 0 or client_id < 0 or client_id >= num_clients:
            logger.error(f"Invalid partitioning parameters: client_id={client_id}, num_clients={num_clients}")
            return self.samples
        
        # Shuffle for better distribution
        import random
        shuffled_samples = self.samples.copy()
        random.Random(42).shuffle(shuffled_samples)  # Fixed seed for reproducibility
        
        samples_per_client = len(shuffled_samples) // num_clients
        start_idx = client_id * samples_per_client
        
        if client_id == num_clients - 1:  # Last client gets remaining samples
            client_samples = shuffled_samples[start_idx:]
        else:
            client_samples = shuffled_samples[start_idx:start_idx + samples_per_client]
        
        logger.info(f"Client {client_id} assigned {len(client_samples)} samples out of {len(self.samples)} total")
        return client_samples
    
    def _resolve_nii_path(self, nii_path: str) -> str:
        """
        Resolve ACDC's special directory structure where .nii "files" are actually directories.
        
        ACDC structure:
        patient065_frame01.nii/         (directory)
        ├── NOR09Gate1.nii             (actual file)
        """
        nii_path_obj = Path(nii_path)
        
        # If it's already a real file, return as-is
        if nii_path_obj.is_file():
            return str(nii_path_obj)
        
        # If it's a directory (ACDC case), look for .nii file inside
        if nii_path_obj.is_dir():
            nii_files = list(nii_path_obj.glob("*.nii"))
            if nii_files:
                # Try to find a valid NIfTI file (not ASCII text)
                for nii_file in nii_files:
                    try:
                        # Quick nibabel test to check if file is valid
                        nib_load(str(nii_file))
                        return str(nii_file)  # Found valid file
                    except Exception:
                        continue
                # If no valid file found, return first one anyway
                return str(nii_files[0])
        
        # Fallback: try .nii.gz format
        nii_gz_path = nii_path_obj.with_suffix('.nii.gz')
        if nii_gz_path.is_file():
            return str(nii_gz_path)
        
        # Nothing found
        return ""
    
    def _load_sample_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Raw sample loading with robust error handling (will be cached by LRU cache)."""
        if idx >= len(self.samples):
            logger.error(f"Index {idx} out of range for {len(self.samples)} samples")
            return self._get_dummy_data()
            
        sample = self.samples[idx]
        
        try:
            # Validate file existence first
            if not Path(sample['image_path']).exists():
                logger.error(f"Image file not found: {sample['image_path']}")
                return self._get_dummy_data()
                
            if not Path(sample['mask_path']).exists():
                logger.error(f"Mask file not found: {sample['mask_path']}")
                return self._get_dummy_data()
            
            # Handle ACDC's special directory structure
            actual_image_path = self._resolve_nii_path(sample['image_path'])
            actual_mask_path = self._resolve_nii_path(sample['mask_path'])
            
            if not actual_image_path or not actual_mask_path:
                logger.error(f"Could not resolve file paths for {sample['patient_id']}")
                return self._get_dummy_data()
            
            # Load with nibabel - improved error handling with proper type safety
            try:
                # Use nib.load directly for better type inference
                image_nii = nib_load(actual_image_path)
                mask_nii = nib_load(actual_mask_path)
            except Exception as load_error:
                logger.error(f"Nibabel loading failed for {sample['patient_id']}: {load_error}")
                return self._get_dummy_data()
            
            # Get data with proper error checking and type safety
            try:
                # Use try/except for robust data extraction to avoid linter type issues
                try:
                    image_data = image_nii.get_fdata()  # type: ignore
                except AttributeError:
                    image_data = image_nii.get_data()  # type: ignore
                    
                try:
                    mask_data = mask_nii.get_fdata()  # type: ignore
                except AttributeError:
                    mask_data = mask_nii.get_data()  # type: ignore
                
                image = np.array(image_data).astype(np.float32)
                mask = np.array(mask_data).astype(np.int64)
            except Exception as data_error:
                logger.error(f"Data extraction failed for {sample['patient_id']}: {data_error}")
                return self._get_dummy_data()
            
            # Validate data dimensions
            if image.ndim not in [2, 3] or mask.ndim not in [2, 3]:
                logger.error(f"Invalid dimensions: image {image.shape}, mask {mask.shape}")
                return self._get_dummy_data()
            
            # Handle 3D by selecting best slice (like Kaggle approach)
            if image.ndim == 3:
                image, mask = self._select_best_slice(image, mask)
            
            # Validate data integrity
            if image.size == 0 or mask.size == 0:
                logger.error(f"Empty data loaded from {sample['image_path']}")
                return self._get_dummy_data()
            
            # Apply preprocessing
            image = self._preprocess_image(image)
            mask = self._preprocess_mask(mask)
            
            return image, mask
            
        except Exception as e:
            logger.error(f"Failed to load sample {idx} ({sample['patient_id']}): {e}")
            return self._get_dummy_data()
    
    def _select_best_slice(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select the slice with most foreground content (non-background)."""
        if image.ndim != 3 or mask.ndim != 3:
            return image, mask
        
        # Find slice with maximum foreground content
        max_foreground = 0
        best_slice = image.shape[2] // 2  # Default to middle slice
        
        for i in range(image.shape[2]):
            foreground_pixels = int(np.sum(mask[:, :, i] > 0))
            if foreground_pixels > max_foreground:
                max_foreground = foreground_pixels
                best_slice = i
        
        return image[:, :, best_slice], mask[:, :, best_slice]
    
    def _get_dummy_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return dummy data to prevent training crash."""
        return (
            np.zeros(self.image_size, dtype=np.float32), 
            np.zeros(self.image_size, dtype=np.int64)
        )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Medical image preprocessing with proper techniques."""
        # Clip outliers using percentiles (medical imaging standard)
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
        
        # Normalize to [0, 1] range
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        
        # Resize using skimage for better quality (like Kaggle code)
        if image.shape != self.image_size:
            from skimage.transform import resize
            image = resize(
                image, self.image_size,
                order=1,  # Bilinear interpolation for images
                preserve_range=True,
                anti_aliasing=True,
                mode='reflect'
            ).astype(np.float32)
        
        # Ensure contiguous array to avoid stride issues
        return np.ascontiguousarray(image, dtype=np.float32)
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Medical mask preprocessing with proper techniques."""
        # Clip to valid class range [0, 3] for ACDC
        mask = np.clip(mask, 0, 3)
        
        # Resize using skimage for better quality (like Kaggle code)
        if mask.shape != self.image_size:
            from skimage.transform import resize
            mask = resize(
                mask, self.image_size,
                order=0,  # Nearest neighbor for masks (preserve labels)
                preserve_range=True,
                anti_aliasing=False,  # No anti-aliasing for discrete labels
                mode='reflect'
            ).astype(np.uint8)
        
        # Ensure contiguous array to avoid stride issues
        return np.ascontiguousarray(mask, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get preprocessed sample with optional augmentation."""
        # Load from cache or disk
        image, mask = self._load_sample_cached(idx)
        
        # Apply augmentation if specified
        if self.augment_fn is not None:
            image, mask = self.augment_fn(image, mask)
        
        # Convert to tensors - FIX: Copy arrays to avoid negative stride issues
        image_tensor = torch.from_numpy(image.copy()).unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.from_numpy(mask.copy())  # [H, W]
        
        return image_tensor, mask_tensor
    
    def get_sample_info(self, idx: int) -> Dict[str, str]:
        """Get sample metadata."""
        return self.samples[idx] if 0 <= idx < len(self.samples) else {}


# Simple augmentation functions
def simple_augment(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple augmentation with basic transforms and improved error handling."""
    try:
        # Random horizontal flip - FIX: Copy arrays to avoid negative stride
        if np.random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        # Random rotation (small angle)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Rotate image
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask.astype(np.float32), M, (w, h), 
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT).astype(np.int64)
        
        # Add slight noise to image only
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image, mask
    except Exception as e:
        logger.warning(f"Augmentation failed: {e}, returning original data")
        return image, mask


def create_simple_dataloader(
    data_dir: str,
    batch_size: int = 4,
    shuffle: bool = True,
    augment: bool = False,
    num_workers: int = 0,  # Changed default to 0 for stability
    client_id: Optional[int] = None,
    num_clients: Optional[int] = None,
    cache_size: int = 50,
    **kwargs
) -> DataLoader:
    """
    Create optimized DataLoader with enhanced error handling.
    
    Args:
        data_dir: Path to ACDC dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        augment: Whether to apply augmentation
        num_workers: Number of worker processes (0 for single-process)
        client_id: Client ID for federated learning
        num_clients: Total number of clients
        cache_size: Number of samples to cache
        **kwargs: Additional arguments for SimpleMedicalDataset
    """
    # Setup augmentation
    augment_fn = simple_augment if augment else None
    
    # Create dataset with error handling
    try:
        dataset = SimpleMedicalDataset(
            data_dir=data_dir,
            augment_fn=augment_fn,
            client_id=client_id,
            num_clients=num_clients,
            cache_size=cache_size,
            **kwargs
        )
        
        if len(dataset) == 0:
            logger.error(f"Empty dataset created from {data_dir}")
            raise ValueError(f"No valid samples found in {data_dir}")
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise
    
    # Create DataLoader with optimal settings
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True and torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=False
        )
        
        logger.info(f"DataLoader created: {len(dataset)} samples, batch_size={batch_size}")
        return dataloader
        
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}")
        raise


# Quick test function
def test_dataset(data_dir: str, num_samples: int = 5):
    """Test dataset functionality with comprehensive validation."""
    print("Testing Enhanced SimpleMedicalDataset...")
    
    try:
        dataset = SimpleMedicalDataset(data_dir, cache_size=10)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("No samples found! Check:")
            print("1. Data directory path is correct")
            print("2. Info.cfg files exist in patient directories")
            print("3. NIfTI files (.nii or .nii.gz) are present")
            return False
        
        # Test loading with detailed info
        successful_loads = 0
        for i in range(min(num_samples, len(dataset))):
            try:
                image, mask = dataset[i]
                info = dataset.get_sample_info(i)
                
                # Validate data quality
                unique_classes = np.unique(mask.numpy())
                foreground_pixels = int(torch.sum(mask > 0).item())
                total_pixels = int(mask.numel())
                foreground_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0.0
                
                print(f"Sample {i}: Image {image.shape}, Mask {mask.shape}")
                print(f"  Patient: {info.get('patient_id', 'unknown')}")
                print(f"  Phase: {info.get('cardiac_phase', 'unknown')}")
                print(f"  Frame: {info.get('frame', 'unknown')}")
                print(f"  Classes: {unique_classes}")
                print(f"  Foreground: {foreground_ratio:.1%}")
                print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
                
                successful_loads += 1
                
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
        
        print(f"\nTest completed! {successful_loads}/{min(num_samples, len(dataset))} samples loaded successfully.")
        
        if successful_loads > 0:
            print("Dataset is ready for training!")
            return True
        else:
            print("Dataset has critical issues!")
            return False
            
    except Exception as e:
        print(f"Dataset test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the dataset if run directly
    import sys
    if len(sys.argv) > 1:
        test_data_dir = sys.argv[1]
    else:
        test_data_dir = "data/raw/ACDC/database/training"
    
    test_dataset(test_data_dir) 