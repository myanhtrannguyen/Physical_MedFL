import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
import os

class B1MapCommonCalculator:
    
    def __init__(self, img_size: int = 256, device: str = 'cuda'):
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.common_b1_map = None
        self.dataset_statistics = {}
        
    def simulate_b1_map_physics_based(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Mô phỏng B1_map dựa trên nguyên lý vật lý MRI và đặc trưng ảnh
        
        Args:
            image_batch: Tensor shape (B, C, H, W)
        Returns:
            b1_maps: Tensor shape (B, 1, H, W)
        """
        batch_size, channels, height, width = image_batch.shape
        device = image_batch.device
        
        # Tạo coordinate grids
        y_coords = torch.arange(height, dtype=torch.float32, device=device)
        x_coords = torch.arange(width, dtype=torch.float32, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        center_y, center_x = height // 2, width // 2
        
        # Distance từ center (RF coil thường đặt ở giữa)
        distance = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_distance = torch.sqrt(torch.tensor(center_x**2 + center_y**2, device=device))
        
        b1_maps = []
        
        for b in range(batch_size):
            # Lấy ảnh của batch hiện tại
            current_image = image_batch[b, 0]  # Shape: (H, W)
            
            # 1. B1 inhomogeneity pattern cơ bản (giảm từ center ra ngoài)
            b1_base = 1.0 - 0.25 * (distance / max_distance)
            
            # 2. Tissue-dependent variations
            # Mô có cường độ cao (như máu) có dielectric constant khác
            image_normalized = current_image / (torch.max(current_image) + 1e-8)
            tissue_factor = 0.85 + 0.3 * image_normalized
            
            # 3. Cardiac-specific adjustments
            # Tim có hình dạng và vị trí đặc biệt
            cardiac_factor = self._get_cardiac_b1_pattern(current_image, height, width, device)
            
            # 4. RF coil loading effects
            # Tải RF phụ thuộc vào phân bố mô
            loading_effect = self._calculate_rf_loading(current_image, distance, device)
            
            # 5. Kết hợp các yếu tố
            b1_map = b1_base * tissue_factor * cardiac_factor * loading_effect
            
            # 6. Thêm realistic noise và constraints
            noise = torch.randn_like(b1_map, device=device) * 0.03
            b1_map = b1_map + noise
            
            # Clip về range thực tế của B1 field (0.4 - 1.3)
            b1_map = torch.clamp(b1_map, 0.4, 1.3)
            
            b1_maps.append(b1_map.unsqueeze(0))  # Add channel dimension
        
        return torch.stack(b1_maps, dim=0)  # Shape: (B, 1, H, W)
    
    def _get_cardiac_b1_pattern(self, image: torch.Tensor, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Tạo B1 pattern đặc biệt cho cardiac imaging
        """
        # Cardiac region thường ở center-left của ảnh
        cardiac_center_y = height // 2
        cardiac_center_x = int(width * 0.4)  # Slightly left of center
        
        y_coords = torch.arange(height, dtype=torch.float32, device=device)
        x_coords = torch.arange(width, dtype=torch.float32, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Distance từ cardiac center
        cardiac_distance = torch.sqrt((x_grid - cardiac_center_x)**2 + (y_grid - cardiac_center_y)**2)
        
        # B1 field mạnh hơn ở cardiac region
        cardiac_enhancement = 1.0 + 0.1 * torch.exp(-cardiac_distance / (width * 0.15))
        
        # Modulate bởi image intensity (cardiac structures có contrast cao)
        image_weight = image / (torch.max(image) + 1e-8)
        cardiac_factor = cardiac_enhancement * (0.9 + 0.2 * image_weight)
        
        return cardiac_factor
    
    def _calculate_rf_loading(self, image: torch.Tensor, distance: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Tính toán RF loading effect dựa trên phân bố mô
        """
        # RF loading tăng khi có nhiều mô (high intensity regions)
        tissue_density = image / (torch.max(image) + 1e-8)
        
        # Loading effect mạnh hơn ở center (gần RF coil)
        loading_base = 1.0 - 0.1 * (distance / torch.max(distance))
        
        # Kết hợp với tissue density
        loading_effect = loading_base * (0.95 + 0.1 * tissue_density)
        
        return loading_effect
    
    def calculate_dataset_common_b1_map(self, 
                                      all_images: torch.Tensor,
                                      use_weighted_average: bool = True,
                                      save_path: Optional[str] = None) -> torch.Tensor:
        print("Calculating common B1 map for entire ACDC dataset...")
        
        num_images = all_images.shape[0]
        batch_size = min(16, num_images)  # Process in batches để tránh memory overflow
        
        all_b1_maps = []
        image_statistics = []
        
        # Process images in batches
        for i in range(0, num_images, batch_size):
            end_idx = min(i + batch_size, num_images)
            batch_images = all_images[i:end_idx].to(self.device)
            
            # Generate B1 maps for current batch
            batch_b1_maps = self.simulate_b1_map_physics_based(batch_images)
            all_b1_maps.append(batch_b1_maps.cpu())
            
            # Collect statistics
            for j in range(batch_images.shape[0]):
                img_stats = {
                    'mean_intensity': torch.mean(batch_images[j]).item(),
                    'std_intensity': torch.std(batch_images[j]).item(),
                    'max_intensity': torch.max(batch_images[j]).item()
                }
                image_statistics.append(img_stats)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + batch_size}/{num_images} images...")
        
        # Concatenate all B1 maps
        all_b1_maps = torch.cat(all_b1_maps, dim=0)  # Shape: (N, 1, H, W)
        
        if use_weighted_average:
            common_b1_map = self._calculate_weighted_average(all_b1_maps, image_statistics)
        else:
            common_b1_map = torch.mean(all_b1_maps, dim=0, keepdim=True)
        
        # Store results
        self.common_b1_map = common_b1_map
        self.dataset_statistics = {
            'num_images': num_images,
            'b1_range': (float(torch.min(common_b1_map)), float(torch.max(common_b1_map))),
            'b1_mean': float(torch.mean(common_b1_map)),
            'b1_std': float(torch.std(common_b1_map)),
            'image_stats': {
                'mean_intensity_avg': np.mean([s['mean_intensity'] for s in image_statistics]),
                'std_intensity_avg': np.mean([s['std_intensity'] for s in image_statistics])
            }
        }
        
        # print(f"Common B1 map calculated successfully!")
        # print(f"  - B1 range: {self.dataset_statistics['b1_range'][0]:.3f} - {self.dataset_statistics['b1_range'][1]:.3f}")
        # print(f"  - B1 mean: {self.dataset_statistics['b1_mean']:.3f}")
        # print(f"  - B1 std: {self.dataset_statistics['b1_std']:.3f}")
        
        # Save if path provided
        if save_path:
            self.save_common_b1_map(save_path)
        
        return common_b1_map
    
    def _calculate_weighted_average(self, all_b1_maps: torch.Tensor, image_stats: list) -> torch.Tensor:
        """
        Tính weighted average, ưu tiên các vùng center và ảnh có contrast cao
        """
        _, _, height, width = all_b1_maps.shape
        
        # Create spatial weights (higher weight for center regions)
        y_coords = torch.arange(height, dtype=torch.float32)
        x_coords = torch.arange(width, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        center_y, center_x = height // 2, width // 2
        spatial_weights = torch.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * (min(height, width) / 4)**2))
        spatial_weights = spatial_weights.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        
        # Create image-wise weights based on contrast
        image_weights = []
        for stats in image_stats:
            # Higher weight for images with good contrast
            contrast_score = stats['std_intensity'] / (stats['mean_intensity'] + 1e-8)
            weight = min(max(contrast_score, 0.5), 2.0)  # Clip between 0.5 and 2.0
            image_weights.append(weight)
        
        image_weights = torch.tensor(image_weights).view(-1, 1, 1, 1)  # Shape: (N, 1, 1, 1)
        
        # Apply weights
        weighted_b1_maps = all_b1_maps * spatial_weights * image_weights
        total_weights = spatial_weights * image_weights
        
        # Calculate weighted average
        common_b1_map = torch.sum(weighted_b1_maps, dim=0, keepdim=True) / torch.sum(total_weights, dim=0, keepdim=True)
        
        return common_b1_map
    
    def get_b1_map_for_batch(self, batch_images: torch.Tensor) -> torch.Tensor:
        """
        Lấy B1_map cho một batch ảnh
        Args:
            batch_images: Tensor shape (B, C, H, W)
        Returns:
            b1_maps: Tensor shape (B, 1, H, W)
        """
        if self.common_b1_map is not None:
            # Sử dụng common B1 map, broadcast cho toàn batch
            batch_size = batch_images.shape[0]
            return self.common_b1_map.expand(batch_size, -1, -1, -1).to(batch_images.device)
        else:
            # Tính B1 map riêng cho batch này
            return self.simulate_b1_map_physics_based(batch_images)
    
    def save_common_b1_map(self, save_path: str):
        """Lưu common B1 map và statistics"""
        if self.common_b1_map is not None:
            save_dict = {
                'common_b1_map': self.common_b1_map,
                'dataset_statistics': self.dataset_statistics,
                'img_size': self.img_size
            }
            torch.save(save_dict, save_path)
            # print(f"Common B1 map saved to: {save_path}")
    
    def load_common_b1_map(self, load_path: str):
        """Load common B1 map từ file đã lưu"""
        if os.path.exists(load_path):
            save_dict = torch.load(load_path, map_location=self.device)
            self.common_b1_map = save_dict['common_b1_map']
            self.dataset_statistics = save_dict['dataset_statistics']
            self.img_size = save_dict.get('img_size', 256)
            # print(f"Common B1 map loaded from: {load_path}")
            # print(f"  - B1 range: {self.dataset_statistics['b1_range'][0]:.3f} - {self.dataset_statistics['b1_range'][1]:.3f}")
        # else:
            # print(f"File not found: {load_path}")

def integrate_b1_map_into_training(X_train_tensor: torch.Tensor, 
                                 X_val_tensor: torch.Tensor, 
                                 X_test_tensor: torch.Tensor,
                                 img_size: int = 256,
                                 device: str = 'cuda') -> B1MapCommonCalculator:
    
    # print("=== Integrating B1 Map Calculator ===")
    
    # Initialize calculator
    b1_calculator = B1MapCommonCalculator(img_size=img_size, device=device)
    
    # Try to load existing common B1 map
    save_path = "acdc_common_b1_map.pth"
    b1_calculator.load_common_b1_map(save_path)
    
    # If not loaded, calculate new one
    if b1_calculator.common_b1_map is None:
        # Combine all images for calculating common B1 map
        all_images = torch.cat([X_train_tensor, X_val_tensor, X_test_tensor], dim=0)
        
        # Calculate common B1 map
        common_b1_map = b1_calculator.calculate_dataset_common_b1_map(
            all_images, 
            use_weighted_average=True,
            save_path=save_path
        )
        
        # print(f"New common B1 map calculated and saved!")
    # else:
    #     print(f"Using existing common B1 map!")
    return b1_calculator

# Hàm thay thế cho việc sử dụng trong training loop
def get_b1_map_for_training(images: torch.Tensor, 
                          b1_calculator: B1MapCommonCalculator) -> torch.Tensor:
    return b1_calculator.get_b1_map_for_batch(images)