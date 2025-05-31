"""
Demo Data Loader for Testing the Experimental Pipeline
Generates synthetic medical image data similar to ACDC/BraTS format
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
import logging


class DemoMedicalDataset(Dataset):
    """
    Demo dataset that generates synthetic medical images and segmentation masks.
    Simulates cardiac MRI data similar to ACDC dataset.
    """
    
    def __init__(self, 
                 num_samples: int = 100,
                 image_size: Tuple[int, int] = (128, 128),
                 num_classes: int = 4,
                 transform=None):
        
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        
        # Generate consistent synthetic data
        np.random.seed(42)
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic cardiac MRI-like data"""
        data = []
        
        for i in range(self.num_samples):
            # Create synthetic cardiac image
            image = self._create_cardiac_image()
            
            # Create corresponding segmentation mask
            mask = self._create_segmentation_mask()
            
            data.append({
                'image': image,
                'mask': mask,
                'patient_id': f"demo_patient_{i:03d}",
                'slice_id': f"slice_{i % 10:02d}"
            })
        
        return data
    
    def _create_cardiac_image(self) -> np.ndarray:
        """Create synthetic cardiac MRI image"""
        h, w = self.image_size
        
        # Start with noise
        image = np.random.normal(0.1, 0.05, (h, w))
        
        # Add heart-like structure
        center_x, center_y = w // 2, h // 2
        
        # Left ventricle (bright circle)
        lv_radius = np.random.uniform(15, 25)
        lv_center_x = center_x + np.random.uniform(-10, 10)
        lv_center_y = center_y + np.random.uniform(-10, 10)
        
        y, x = np.ogrid[:h, :w]
        lv_mask = ((x - lv_center_x)**2 + (y - lv_center_y)**2) <= lv_radius**2
        image[lv_mask] = np.random.uniform(0.7, 0.9)
        
        # Right ventricle (medium circle)
        rv_radius = np.random.uniform(12, 20)
        rv_center_x = center_x + np.random.uniform(20, 40)
        rv_center_y = center_y + np.random.uniform(-15, 15)
        
        rv_mask = ((x - rv_center_x)**2 + (y - rv_center_y)**2) <= rv_radius**2
        image[rv_mask] = np.random.uniform(0.5, 0.7)
        
        # Myocardium (ring around LV)
        myo_outer = lv_radius + np.random.uniform(8, 15)
        myo_mask = (((x - lv_center_x)**2 + (y - lv_center_y)**2) <= myo_outer**2) & (~lv_mask)
        image[myo_mask] = np.random.uniform(0.3, 0.5)
        
        # Add some texture and noise
        texture = np.random.normal(0, 0.02, (h, w))
        image = np.clip(image + texture, 0, 1)
        
        return image.astype(np.float32)
    
    def _create_segmentation_mask(self) -> np.ndarray:
        """Create segmentation mask corresponding to the cardiac image"""
        h, w = self.image_size
        mask = np.zeros((h, w), dtype=np.int64)
        
        center_x, center_y = w // 2, h // 2
        
        # Generate same structures as image
        lv_radius = np.random.uniform(15, 25)
        lv_center_x = center_x + np.random.uniform(-10, 10)
        lv_center_y = center_y + np.random.uniform(-10, 10)
        
        rv_radius = np.random.uniform(12, 20)
        rv_center_x = center_x + np.random.uniform(20, 40)
        rv_center_y = center_y + np.random.uniform(-15, 15)
        
        myo_outer = lv_radius + np.random.uniform(8, 15)
        
        y, x = np.ogrid[:h, :w]
        
        # Class 0: Background (default)
        # Class 1: Right Ventricle
        rv_mask = ((x - rv_center_x)**2 + (y - rv_center_y)**2) <= rv_radius**2
        mask[rv_mask] = 1
        
        # Class 2: Myocardium
        lv_mask = ((x - lv_center_x)**2 + (y - lv_center_y)**2) <= lv_radius**2
        myo_mask = (((x - lv_center_x)**2 + (y - lv_center_y)**2) <= myo_outer**2) & (~lv_mask)
        mask[myo_mask] = 2
        
        # Class 3: Left Ventricle
        mask[lv_mask] = 3
        
        return mask
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        
        image = sample['image']
        mask = sample['mask']
        
        # Convert to torch tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).long()
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask


def create_demo_dataloader(batch_size: int = 4,
                          num_samples: int = 100,
                          image_size: Tuple[int, int] = (128, 128),
                          num_classes: int = 4,
                          shuffle: bool = True,
                          **kwargs) -> DataLoader:
    """
    Create demo data loader for testing the experimental pipeline.
    
    Args:
        batch_size: Batch size for the dataloader
        num_samples: Number of synthetic samples to generate
        image_size: Size of synthetic images (H, W)
        num_classes: Number of segmentation classes
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        DataLoader with synthetic medical data
    """
    
    dataset = DemoMedicalDataset(
        num_samples=num_samples,
        image_size=image_size,
        num_classes=num_classes
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )
    
    return dataloader


def create_demo_research_dataloader(dataset_type: str = 'demo',
                                   data_dir: str = None,
                                   batch_size: int = 4,
                                   shuffle: bool = True,
                                   augment: bool = False,
                                   **kwargs) -> DataLoader:
    """
    Create demo research dataloader compatible with the existing interface.
    
    Args:
        dataset_type: Type of dataset (ignored, always creates demo data)
        data_dir: Data directory (ignored for demo)
        batch_size: Batch size
        shuffle: Whether to shuffle
        augment: Whether to augment (ignored for demo)
        **kwargs: Additional arguments
        
    Returns:
        Demo dataloader
    """
    
    logging.info(f"Creating demo dataloader (batch_size={batch_size}, shuffle={shuffle})")
    
    return create_demo_dataloader(
        batch_size=batch_size,
        num_samples=kwargs.get('num_samples', 100),
        shuffle=shuffle
    )


# Federated learning demo setup
def create_demo_federated_loaders(num_clients: int = 3,
                                 samples_per_client: int = 50,
                                 batch_size: int = 4,
                                 **kwargs) -> Dict[str, DataLoader]:
    """
    Create demo federated data loaders.
    
    Args:
        num_clients: Number of federated clients
        samples_per_client: Samples per client
        batch_size: Batch size per client
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of client_id -> DataLoader
    """
    
    client_loaders = {}
    
    for client_id in range(num_clients):
        # Create different data distributions for each client
        np.random.seed(42 + client_id)  # Different seed per client
        
        loader = create_demo_dataloader(
            batch_size=batch_size,
            num_samples=samples_per_client,
            shuffle=True
        )
        
        client_loaders[f"client_{client_id}"] = loader
    
    logging.info(f"Created {num_clients} demo federated data loaders")
    return client_loaders


if __name__ == "__main__":
    # Test the demo data loader
    print("Testing demo data loader...")
    
    # Create demo dataloader
    loader = create_demo_dataloader(batch_size=2, num_samples=10)
    
    # Test a batch
    for batch_idx, (images, masks) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Mask classes: {torch.unique(masks)}")
        
        if batch_idx >= 2:  # Only show first few batches
            break
    
    print("Demo data loader test completed!") 