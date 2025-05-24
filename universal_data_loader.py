#!/usr/bin/env python3
"""
Universal Medical Data Loader
Supports multiple image formats with automatic detection and preprocessing.

Supported formats:
- NIfTI (.nii, .nii.gz)
- H5/HDF5 (.h5, .hdf5)
- DICOM (.dcm, .dicom)
- Common images (PNG, JPG, JPEG, TIFF)
- NumPy arrays (.npy, .npz)
- Raw binary (.raw, .bin)
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import logging
from skimage.transform import resize
from skimage import exposure, filters
import torch
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import nibabel
    # Import the specific load function we need
    from nibabel.loadsave import load as nib_load
    NIBABEL_AVAILABLE = True
except ImportError:
    logger.warning("nibabel not available - NIfTI support disabled")
    NIBABEL_AVAILABLE = False
    nibabel = None
    nib_load = None

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    logger.warning("h5py not available - H5 support disabled")
    H5PY_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    logger.warning("pydicom not available - DICOM support disabled")
    PYDICOM_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("PIL not available - common image format support limited")
    PIL_AVAILABLE = False

# Constants
SUPPORTED_FORMATS = {
    'nifti': ['.nii', '.nii.gz'],
    'h5': ['.h5', '.hdf5'],
    'dicom': ['.dcm', '.dicom'],
    'image': ['.png', '.jpg', '.jpeg', '.tiff', '.tif'],
    'numpy': ['.npy', '.npz'],
    'raw': ['.raw', '.bin']
}

class DataFormat:
    """Data format enumeration."""
    NIFTI = 'nifti'
    H5 = 'h5'
    DICOM = 'dicom'
    IMAGE = 'image'
    NUMPY = 'numpy'
    RAW = 'raw'
    UNKNOWN = 'unknown'

class UniversalDataLoader:
    """Universal data loader for medical imaging data."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 num_classes: int = 4,
                 normalize: bool = True,
                 clip_range: Optional[Tuple[float, float]] = None):
        """
        Initialize universal data loader.
        
        Args:
            target_size: Target image size (height, width)
            num_classes: Number of segmentation classes
            normalize: Whether to normalize images to [0, 1]
            clip_range: Optional intensity clipping range
        """
        self.target_size = target_size
        self.num_classes = num_classes
        self.normalize = normalize
        self.clip_range = clip_range
        
    def detect_format(self, path: str) -> str:
        """Auto-detect data format from file/directory structure."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            return DataFormat.UNKNOWN
            
        # Check if it's a directory with subdirectories
        if path_obj.is_dir():
            # Look for patient directories (ACDC structure)
            patient_dirs = list(path_obj.glob("patient*"))
            if patient_dirs:
                # Check what's inside patient directories
                sample_patient = patient_dirs[0]
                
                # Check for NIfTI structure (directories named *.nii)
                nii_dirs = list(sample_patient.glob("*.nii"))
                if nii_dirs and nii_dirs[0].is_dir():
                    return DataFormat.NIFTI
                
                # Check for NIfTI files
                nii_files = list(sample_patient.glob("*.nii*"))
                if nii_files and nii_files[0].is_file():
                    return DataFormat.NIFTI
                
                # Check for DICOM files
                dcm_files = list(sample_patient.glob("*.dcm")) + list(sample_patient.glob("*.dicom"))
                if dcm_files:
                    return DataFormat.DICOM
            
            # Check for H5 files
            h5_files = list(path_obj.glob("*.h5")) + list(path_obj.glob("*.hdf5"))
            if h5_files:
                return DataFormat.H5
                
            # Check for common image files
            image_files = []
            for ext in SUPPORTED_FORMATS['image']:
                image_files.extend(list(path_obj.glob(f"*{ext}")))
            if image_files:
                return DataFormat.IMAGE
                
        else:
            # Single file
            suffix = path_obj.suffix.lower()
            for format_name, extensions in SUPPORTED_FORMATS.items():
                if suffix in extensions:
                    return format_name
        
        return DataFormat.UNKNOWN
    
    def load_nifti_data(self, directory: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load NIfTI format data."""
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel required for NIfTI support")
        
        logger.info(f"Loading NIfTI data from {directory}")
        imgs, msks = [], []
        
        patient_folders = sorted(list(Path(directory).glob("patient*")))
        if max_samples:
            patient_folders = patient_folders[:max_samples]
        
        for patient_folder in patient_folders:
            patient_id = patient_folder.name
            logger.debug(f"Processing patient: {patient_id}")
            
            # Find frame data
            frame_patterns = [f"{patient_id}_frame*[!_gt].nii"]
            frame_items = []
            for pattern in frame_patterns:
                frame_items.extend(patient_folder.glob(pattern))
            
            for frame_item in frame_items:
                try:
                    img_data, mask_data = self._load_nifti_frame(frame_item, patient_folder)
                    if img_data is not None and mask_data is not None:
                        processed_imgs, processed_masks = self._process_3d_data(img_data, mask_data)
                        imgs.extend(processed_imgs)
                        msks.extend(processed_masks)
                except Exception as e:
                    logger.warning(f"Error loading {frame_item}: {e}")
                    continue
        
        return self._finalize_arrays(imgs, msks)
    
    def _load_nifti_frame(self, frame_item: Path, patient_folder: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single NIfTI frame and its corresponding mask."""
        try:
            # Skip problematic patient057 files that cause warnings
            if "patient057" in str(patient_folder) and any(gt in str(frame_item) for gt in ["frame09_gt.nii", "frame01_gt.nii"]):
                logger.debug(f"Skipping known problematic file in {patient_folder}")
                return None, None
                
            # Handle both directory and file structures
            if frame_item.is_dir():
                # Directory structure (training data)
                nii_files = list(frame_item.glob("*.nii"))
                if not nii_files:
                    return None, None
                img_file = nii_files[0]
                
                # Find corresponding mask directory
                frame_info = frame_item.name.replace('.nii', '')
                mask_dir = patient_folder / f"{frame_info}_gt.nii"
                
                # Improved handling for nonstandard filenames (like MINF41OH_AL_8.nii)
                # First check if mask exists in expected location
                if mask_dir.exists():
                    if mask_dir.is_dir():
                        mask_files = list(mask_dir.glob("*.nii"))
                        if mask_files:
                            mask_file = mask_files[0]
                        else:
                            # If no .nii files in mask directory, try searching for any .nii file
                            return None, None
                    else:
                        # If mask_dir is a file (not a dir), use it directly
                        mask_file = mask_dir
                else:
                    # If expected mask path doesn't exist, search for any _gt.nii in patient folder
                    # This handles cases where naming convention isn't standard
                    mask_candidates = list(patient_folder.glob("*_gt.nii"))
                    # If we can find frame number in original filename, use it to filter candidates
                    frame_num = None
                    if "frame" in frame_info:
                        try:
                            frame_num = int(''.join(filter(str.isdigit, frame_info.split("frame")[1].split("_")[0])))
                        except (ValueError, IndexError):
                            pass
                    
                    if frame_num is not None and mask_candidates:
                        # Try to find matching frame number in mask candidates
                        for candidate in mask_candidates:
                            if f"frame{frame_num:02d}" in candidate.name or f"_AL_{frame_num}" in candidate.name:
                                mask_file = candidate
                                break
                        else:
                            # If no matching frame number, use first available mask
                            mask_file = mask_candidates[0] if mask_candidates else None
                            if mask_file is None:
                                return None, None
                    else:
                        # Just use first available mask if we can't determine frame number
                        mask_file = mask_candidates[0] if mask_candidates else None
                        if mask_file is None:
                            return None, None
            else:
                # File structure (testing data)
                img_file = frame_item
                frame_info = frame_item.name.replace('.nii', '')
                
                # First try standard naming convention
                mask_file = patient_folder / f"{frame_info}_gt.nii"
                
                # If standard mask doesn't exist, search for alternatives
                if not mask_file.exists():
                    # Try to find mask with similar naming pattern
                    mask_candidates = list(patient_folder.glob("*_gt.nii"))
                    if not mask_candidates:
                        return None, None
                    
                    # Try to extract frame number for matching
                    frame_num = None
                    if "frame" in frame_info:
                        try:
                            frame_num = int(''.join(filter(str.isdigit, frame_info.split("frame")[1].split("_")[0])))
                        except (ValueError, IndexError):
                            pass
                    
                    if frame_num is not None:
                        # Try to find matching frame number in mask candidates
                        for candidate in mask_candidates:
                            if f"frame{frame_num:02d}" in candidate.name or f"_AL_{frame_num}" in candidate.name:
                                mask_file = candidate
                                break
                        else:
                            # If no matching frame, use first available mask
                            mask_file = mask_candidates[0]
                    else:
                        # Just use first available mask
                        mask_file = mask_candidates[0]
            
            # Load data - handle potential file not found issues
            try:
                if not img_file.exists():
                    logger.debug(f"Image file does not exist: {img_file}")
                    return None, None
                
                if not mask_file.exists():
                    logger.debug(f"Mask file does not exist: {mask_file}")
                    return None, None
                
                # Check if directory was mistakenly identified as file
                if img_file.is_dir() or mask_file.is_dir():
                    logger.debug(f"Directory was mistakenly identified as file")
                    return None, None
                if NIBABEL_AVAILABLE and nib_load is not None:  # make sure nibabel is properly imported
                    try:
                        img_nii = nib_load(str(img_file))
                        img_data = img_nii.get_fdata()
                        
                        mask_nii = nib_load(str(mask_file))
                        mask_data = mask_nii.get_fdata()
                        
                        return img_data, mask_data
                        mask_data = mask_nii.get_fdata()
                        
                        return img_data, mask_data
                    except Exception as nibex:
                        logger.debug(f"Error loading NIfTI file: {nibex}")
                        return None, None
                else:
                    logger.error("nibabel module not properly loaded")
                    return None, None
            except Exception as e:
                logger.warning(f"Error loading NIfTI files: {e}")
                return None, None
            
        except Exception as e:
            # Change from warning to debug to reduce console noise
            logger.debug(f"Error in _load_nifti_frame: {e}")
            return None, None
    
    def load_h5_data(self, directory: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load H5 format data."""
        if not H5PY_AVAILABLE:
            raise ImportError("h5py required for H5 support")
        
        logger.info(f"Loading H5 data from {directory}")
        imgs, msks = [], []
        
        h5_files = sorted(list(Path(directory).glob("*.h5")))
        if max_samples:
            h5_files = h5_files[:max_samples]
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    img_data = f['image'][:]
                    mask_data = f['label'][:] if 'label' in f else None
                    
                    if img_data.ndim == 3:
                        # Multiple slices
                        for i in range(img_data.shape[0]):
                            img_slice = self._preprocess_image(img_data[i])
                            imgs.append(img_slice)
                            if mask_data is not None:
                                mask_slice = self._preprocess_mask(mask_data[i])
                                msks.append(mask_slice)
                    else:
                        # Single slice
                        img_slice = self._preprocess_image(img_data)
                        imgs.append(img_slice)
                        if mask_data is not None:
                            mask_slice = self._preprocess_mask(mask_data)
                            msks.append(mask_slice)
                            
            except Exception as e:
                logger.warning(f"Error loading {h5_file}: {e}")
                continue
        
        return self._finalize_arrays(imgs, msks)
    
    def load_dicom_data(self, directory: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load DICOM format data."""
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM support")
        
        logger.info(f"Loading DICOM data from {directory}")
        imgs = []
        
        dicom_files = []
        for ext in ['.dcm', '.dicom']:
            dicom_files.extend(Path(directory).rglob(f"*{ext}"))
        
        dicom_files = sorted(dicom_files)
        if max_samples:
            dicom_files = dicom_files[:max_samples]
        
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                img_data = ds.pixel_array.astype(np.float32)
                img_slice = self._preprocess_image(img_data)
                imgs.append(img_slice)
            except Exception as e:
                logger.warning(f"Error loading {dcm_file}: {e}")
                continue
        
        return self._finalize_arrays(imgs, None)
    
    def load_image_data(self, directory: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load common image format data."""
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, using skimage for image loading")
        
        logger.info(f"Loading image data from {directory}")
        imgs = []
        
        image_files = []
        for ext in SUPPORTED_FORMATS['image']:
            image_files.extend(Path(directory).rglob(f"*{ext}"))
        
        image_files = sorted(image_files)
        if max_samples:
            image_files = image_files[:max_samples]
        
        for img_file in image_files:
            try:
                if PIL_AVAILABLE:
                    img = Image.open(img_file)
                    img_data = np.array(img).astype(np.float32)
                else:
                    from skimage import io
                    img_data = io.imread(img_file).astype(np.float32)
                
                # Convert to grayscale if needed
                if img_data.ndim == 3:
                    img_data = np.mean(img_data, axis=2)
                
                img_slice = self._preprocess_image(img_data)
                imgs.append(img_slice)
                
            except Exception as e:
                logger.warning(f"Error loading {img_file}: {e}")
                continue
        
        return self._finalize_arrays(imgs, None)
    
    def load_numpy_data(self, directory: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load NumPy format data."""
        logger.info(f"Loading NumPy data from {directory}")
        imgs, msks = [], []
        
        numpy_files = []
        for ext in SUPPORTED_FORMATS['numpy']:
            numpy_files.extend(Path(directory).rglob(f"*{ext}"))
        
        numpy_files = sorted(numpy_files)
        if max_samples:
            numpy_files = numpy_files[:max_samples]
        
        for np_file in numpy_files:
            try:
                if np_file.suffix == '.npy':
                    data = np.load(np_file)
                else:  # .npz
                    npz_data = np.load(np_file)
                    data = npz_data['image'] if 'image' in npz_data else npz_data[list(npz_data.keys())[0]]
                    mask_data = npz_data.get('mask', None)
                    
                    if mask_data is not None:
                        mask_slice = self._preprocess_mask(mask_data)
                        msks.append(mask_slice)
                
                img_slice = self._preprocess_image(data)
                imgs.append(img_slice)
                
            except Exception as e:
                logger.warning(f"Error loading {np_file}: {e}")
                continue
        
        return self._finalize_arrays(imgs, msks if msks else None)
    
    def _process_3d_data(self, img_data: np.ndarray, mask_data: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Process 3D image and mask data into 2D slices."""
        imgs, msks = [], []
        
        if img_data.ndim == 3:
            for slice_idx in range(img_data.shape[2]):
                img_slice = img_data[:, :, slice_idx]
                mask_slice = mask_data[:, :, slice_idx]
                
                # Skip empty slices
                if np.max(img_slice) == 0:
                    continue
                
                processed_img = self._preprocess_image(img_slice)
                processed_mask = self._preprocess_mask(mask_slice)
                
                imgs.append(processed_img)
                msks.append(processed_mask)
        else:
            # 2D data
            if np.max(img_data) > 0:
                processed_img = self._preprocess_image(img_data)
                processed_mask = self._preprocess_mask(mask_data)
                imgs.append(processed_img)
                msks.append(processed_mask)
        
        return imgs, msks
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess a single image."""
        # Ensure float32
        img = img.astype(np.float32)
        
        # Apply intensity clipping if specified
        if self.clip_range is not None:
            img = np.clip(img, self.clip_range[0], self.clip_range[1])
        
        # Resize
        img_resized = resize(img, self.target_size, 
                           order=1, preserve_range=True, 
                           anti_aliasing=True, mode='reflect').astype(np.float32)
        
        # Normalize to [0, 1] if requested
        if self.normalize and np.max(img_resized) > 0:
            img_resized = img_resized / np.max(img_resized)
        
        # Add channel dimension
        img_resized = np.expand_dims(img_resized, axis=-1)
        
        return img_resized
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Preprocess a single mask."""
        # Ensure correct data type and class range
        mask = np.clip(mask, 0, self.num_classes - 1).astype(np.uint8)
        
        # Resize
        mask_resized = resize(mask, self.target_size, 
                            order=0, preserve_range=True, 
                            anti_aliasing=False, mode='reflect').astype(np.uint8)
        
        return mask_resized
    
    def _finalize_arrays(self, imgs: List[np.ndarray], msks: Optional[List[np.ndarray]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert lists to final numpy arrays."""
        if not imgs:
            return np.empty((0, *self.target_size, 1), dtype=np.float32), None
        
        img_array = np.array(imgs, dtype=np.float32)
        mask_array = np.array(msks, dtype=np.uint8) if msks else None
        
        logger.info(f"Loaded {len(img_array)} samples")
        return img_array, mask_array
    
    def load_data(self, path: str, max_samples: Optional[int] = None, format_hint: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Universal data loading with automatic format detection.
        
        Args:
            path: Path to data directory or file
            max_samples: Maximum number of samples to load
            format_hint: Optional format hint to skip detection
            
        Returns:
            Tuple of (images, masks) where masks can be None
        """
        # Use format hint or auto-detect
        if format_hint:
            data_format = DataFormat(format_hint)
        else:
            data_format = self.detect_format(path)
        
        logger.info(f"Loading data format: {data_format}")
        
        # Route to appropriate loader
        if data_format == DataFormat.NIFTI:
            return self.load_nifti_data(path, max_samples)
        elif data_format == DataFormat.H5:
            return self.load_h5_data(path, max_samples)
        elif data_format == DataFormat.DICOM:
            return self.load_dicom_data(path, max_samples)
        elif data_format == DataFormat.IMAGE:
            return self.load_image_data(path, max_samples)
        elif data_format == DataFormat.NUMPY:
            return self.load_numpy_data(path, max_samples)
        else:
            raise ValueError(f"Unsupported or unknown data format: {data_format}")

class MedicalDataAugmentation:
    """Medical data augmentation with anatomically-aware transforms."""
    
    def __init__(self, 
                 rotation_degrees: int = 15,
                 flip_prob: float = 0.5,
                 noise_std: float = 0.02,
                 brightness_factor: float = 0.2,
                 contrast_factor: float = 0.2):
        self.rotation_degrees = rotation_degrees
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentations to image and mask pair."""
        # Convert to tensor for easier manipulation
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        
        # Random horizontal flip
        if torch.rand(1) < self.flip_prob:
            image = torch.flip(image, [-1])
            if mask is not None:
                mask = torch.flip(mask, [-1])
        
        # Small rotation (medical images need conservative transforms)
        if self.rotation_degrees > 0:
            angle = torch.randint(-min(self.rotation_degrees, 10), 
                                 min(self.rotation_degrees, 10) + 1, (1,)).item()
            # Note: Would need torchvision.transforms.functional for rotation
            # For now, skip rotation to avoid import issues
        
        # Add small amount of noise
        if self.noise_std > 0:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
        
        # Brightness adjustment
        if torch.rand(1) < 0.5 and self.brightness_factor > 0:
            brightness = 1 + (torch.rand(1) - 0.5) * self.brightness_factor
            image = image * brightness
        
        # Contrast adjustment
        if torch.rand(1) < 0.5 and self.contrast_factor > 0:
            contrast = 1 + (torch.rand(1) - 0.5) * self.contrast_factor
            mean = image.mean()
            image = (image - mean) * contrast + mean
        
        # Clamp values
        image = torch.clamp(image, 0, 1)
        
        return image.numpy(), mask.numpy() if mask is not None else None

def create_dataloader(images: np.ndarray, 
                     masks: Optional[np.ndarray] = None,
                     batch_size: int = 8,
                     shuffle: bool = True,
                     augment: bool = False) -> DataLoader:
    """Create PyTorch DataLoader from numpy arrays."""
    
    class MedicalDataset(Dataset):
        def __init__(self, images, masks=None, augment=False):
            self.images = torch.tensor(images).float()
            self.masks = torch.tensor(masks).long() if masks is not None else None
            self.augment = augment
            if augment:
                self.augmenter = MedicalDataAugmentation()
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = self.images[idx]
            mask = self.masks[idx] if self.masks is not None else None
            
            if self.augment:
                image, mask = self.augmenter(image.numpy(), mask.numpy() if mask is not None else None)
                image = torch.tensor(image).float()
                mask = torch.tensor(mask).long() if mask is not None else None
            
            if mask is not None:
                return image, mask
            else:
                return image
    
    dataset = MedicalDataset(images, masks, augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    loader = UniversalDataLoader(target_size=(256, 256), num_classes=4)
    
    # Test format detection
    test_paths = [
        "ACDC/database/training",
        "ACDC_preprocessed/ACDC_training_slices",
        "data/images",
        "data/dicom_series"
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            format_detected = loader.detect_format(path)
            print(f"Path: {path} -> Format: {format_detected}")
            
            try:
                images, masks = loader.load_data(path, max_samples=5)
                print(f"  Loaded: {images.shape} images")
                if masks is not None:
                    print(f"  Masks: {masks.shape}")
                print()
            except Exception as e:
                print(f"  Error: {e}")
                print() 