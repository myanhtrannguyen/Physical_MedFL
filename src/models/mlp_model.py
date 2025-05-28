import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
from skimage.transform import resize
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Import unified data system
try:
    from src.data import (
        UnifiedFederatedLoader,
        create_acdc_loader,
        create_brats_loader,
        create_multi_medical_loader,
        ACDCUnifiedDataset,
        BraTS2020UnifiedDataset,
        MedicalImagePreprocessor,
        DataAugmentation
    )
    UNIFIED_DATA_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path when running as script
        import sys
        sys.path.append('.')
        from src.data import (
            UnifiedFederatedLoader,
            create_acdc_loader,
            create_brats_loader,
            create_multi_medical_loader,
            ACDCUnifiedDataset,
            BraTS2020UnifiedDataset,
            MedicalImagePreprocessor,
            DataAugmentation
        )
        UNIFIED_DATA_AVAILABLE = True
    except ImportError:
        print("Warning: Unified data system not available")
        UNIFIED_DATA_AVAILABLE = False

# --- Configuration ---
NUM_EPOCHS_CENTRALIZED = 50 
NUM_CLASSES = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
BATCH_SIZE = 8 

# --- Standard Convolutional Block ---
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# --- ePURE Implementation (Provided) ---
class ePURE(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1) # Ensure output is 1 channel for noise profile
        )

    def forward(self, x):
        x_float = x.float()

        # Estimate a base noise map from the input features
        noise_map_raw = self.conv(x_float) # Output is [B, 1, H, W]

        # Simple approach: just output the learned map directly.
        # The adaptive smoothing uses sigmoid, so the network learns to output values
        # that sigmoid can map to appropriate blending weights.
        noise_map = noise_map_raw # [B, 1, H, W]

        return noise_map # Noise profile estimate (1 channel)
    
import torchvision.transforms.functional as TF
# --- Adaptive Spline Smoothing Implementation (Provided) ---
def adaptive_spline_smoothing(x, noise_profile, kernel_size=5, sigma=1.0):
    """
    Áp dụng làm mịn thích nghi dựa trên noise_profile
    - x: Ảnh đầu vào hoặc feature map [B, C, H, W]
    - noise_profile: Bản đồ nhiễu [B, 1, H, W] (giá trị từ 0 đến 1)
    - kernel_size/sigma: Tham số làm mịn Gaussian
    """
    # Ensure input is float for convolution
    x_float = x.float()

    # Ensure noise_profile is float and 1 channel
    noise_profile_float = noise_profile.float()
    if noise_profile_float.size(1) != 1:
         print(f"Warning: Noise profile expected 1 channel but got {noise_profile_float.size(1)}. Using first channel.")
         noise_profile_float = noise_profile_float[:, :1, :, :]


    # Bước 1: Làm mịn ảnh bằng Gaussian blur
    # Apply Gaussian blur channel-wise
    # kernel_size can be a single int or a tuple (h, w). sigma same.
    # Ensure kernel_size is a tuple if needed, or check F.gaussian_blur docs.
    # F.gaussian_blur expects kernel_size as a tuple of ints (h, w).
    # If kernel_size is an int, it uses that for both dims.
    if isinstance(kernel_size, int):
        kernel_size_tuple = (kernel_size, kernel_size)
    else:
        kernel_size_tuple = kernel_size

    if isinstance(sigma, (int, float)):
         sigma_tuple = (float(sigma), float(sigma))
    else:
         sigma_tuple = sigma

    # Ensure sigma values are positive to avoid issues
    sigma_tuple = tuple(max(0.1, s) for s in sigma_tuple) # Add small epsilon

    smoothed = TF.gaussian_blur(x_float, kernel_size=list(kernel_size_tuple), sigma=list(sigma_tuple))

    # Bước 2: Chuẩn hóa noise_profile (sigmoid) và mở rộng cho đúng số kênh
    # Sigmoid ensures blending weights are between 0 and 1
    # A higher noise_profile value should lead to *more* smoothing.
    # So, blending_weights = noise_profile (after sigmoid)
    blending_weights = torch.sigmoid(noise_profile_float) # [B, 1, H, W]

    # Expand blending_weights to match the number of channels in x
    blending_weights = blending_weights.repeat(1, x_float.size(1), 1, 1) # [B, C, H, W]

    # Ensure dimensions match for blending
    assert blending_weights.shape == x_float.shape, f"Blending weights shape {blending_weights.shape} does not match input shape {x_float.shape}"

    # Bước 3: Trộn ảnh gốc và ảnh đã làm mịn
    # Output = (1 - alpha) * Original + alpha * Smoothed
    # where alpha = blending_weights
    weighted_sum = x_float * (1 - blending_weights) + smoothed * blending_weights

    return weighted_sum

def quantum_noise_injection(features, T=1.0, pauli_prob={'X': 0.00096, 'Y': 0.00096, 'Z': 0.00096, 'None': 0.99712}):
    """
    Áp dụng nhiễu lượng tử dựa trên cơ chế Pauli Noise Injection cho dữ liệu ảnh MRI.
    
    Args:
        features (torch.Tensor): Tensor đầu vào dạng (batch_size, channels, height, width).
        T (float): Hệ số nhiễu, thường trong khoảng [0.5, 1.5].
        pauli_prob (dict): Phân phối xác suất cho các cổng Pauli (X, Y, Z, None).
    
    Returns:
        torch.Tensor: Tensor đầu ra với nhiễu lượng tử được áp dụng.
    """
    features_float = features.float()
    
    if features_float.dim() < 4 or features_float.size(2) < 2 or features_float.size(3) < 2:
        print("Warning: Features too small for quantum noise injection.")
        return features_float

    try:
        # Đảm bảo tensor ở trên thiết bị đúng
        device = features_float.device
        
        # Chuẩn hóa xác suất Pauli với hệ số T
        scaled_prob = {
            'X': pauli_prob['X'] * T,
            'Y': pauli_prob['Y'] * T,
            'Z': pauli_prob['Z'] * T,
            'None': 1.0 - (pauli_prob['X'] + pauli_prob['Y'] + pauli_prob['Z']) * T
        }
        
        # Tạo mặt nạ ngẫu nhiên để chọn cổng Pauli
        batch_size, channels, height, width = features_float.shape
        pauli_choices = ['X', 'Y', 'Z', 'None']
        probabilities = [scaled_prob['X'], scaled_prob['Y'], scaled_prob['Z'], scaled_prob['None']]
        choice_tensor = torch.multinomial(
            torch.tensor(probabilities, device=device),
            batch_size * channels * height * width,
            replacement=True
        ).view(batch_size, channels, height, width)
        
        # Khởi tạo tensor đầu ra
        noisy_features = features_float.clone()
        
        # Áp dụng cổng Pauli
        for i, pauli in enumerate(pauli_choices):
            mask = (choice_tensor == i)
            if pauli == 'X':
                # Cổng Pauli X: Lật giá trị pixel (giả sử giá trị đã chuẩn hóa trong [0, 1])
                noisy_features[mask] = 1.0 - noisy_features[mask]
            elif pauli == 'Y':
                # Cổng Pauli Y: Kết hợp lật bit và thêm nhiễu ngẫu nhiên
                noisy_features[mask] = 1.0 - noisy_features[mask] + 0.1 * torch.randn_like(noisy_features[mask], device=device)
            elif pauli == 'Z':
                # Cổng Pauli Z: Đổi dấu giá trị pixel
                noisy_features[mask] = -noisy_features[mask]
            # 'None': Giữ nguyên giá trị
            
        # Đảm bảo giá trị pixel nằm trong phạm vi [0, 1]
        noisy_features = torch.clamp(noisy_features, 0.0, 1.0)
        
        return noisy_features
    
    except RuntimeError as e:
        print(f"Quantum noise injection failed: {e}. Returning original features.")
        return features_float
    
# --- Model Components (U-Net based) ---
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)
        self.noise_estimator = ePURE(in_channels=in_channels)

    def forward(self, x):
        noise_profile = self.noise_estimator(x)
        x_smoothed = adaptive_spline_smoothing(x, noise_profile)
        x = self.conv_block1(x_smoothed)
        x = self.conv_block2(x)
        return x
    
class MaxwellSolver(nn.Module):
    def __init__(self, in_channels, hidden_dim=32):
        super(MaxwellSolver, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1))
        omega, mu_0, eps_0 = 2 * np.pi * 42.58e6, 4 * np.pi * 1e-7, 8.854187817e-12
        self.k0 = torch.tensor(omega * np.sqrt(mu_0 * eps_0), dtype=torch.float32)

    def forward(self, x):
        eps_sigma_map = self.encoder(x)
        return eps_sigma_map[:, 0:1, :, :], eps_sigma_map[:, 1:2, :, :]

    def compute_helmholtz_residual(self, b1_map, eps, sigma):
        self.k0 = self.k0.to(b1_map.device)
        omega = 2 * np.pi * 42.58e6
        b1_map_complex = torch.complex(b1_map, torch.zeros_like(b1_map)) if not b1_map.is_complex() else b1_map
        eps_r, sig_r = eps.to(b1_map_complex.device), sigma.to(b1_map_complex.device)
        size = b1_map_complex.shape[2:]
        up_eps = F.interpolate(eps_r, size=size, mode='bilinear', align_corners=False)
        up_sig = F.interpolate(sig_r, size=size, mode='bilinear', align_corners=False)
        eps_c = torch.complex(up_eps, -up_sig / omega)
        lap_b1 = self._laplacian_2d(b1_map_complex)
        res = lap_b1 + (self.k0 ** 2) * eps_c * b1_map_complex
        return res.real ** 2 + res.imag ** 2

    def _laplacian_2d(self, x_complex):
        k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=x_complex.device).reshape(1,1,3,3)
        # Handle cases where real or imag part might have 0 channels if x_complex is purely real/imag
        groups_real = x_complex.real.size(1) if x_complex.real.size(1) > 0 else 1
        groups_imag = x_complex.imag.size(1) if x_complex.imag.size(1) > 0 else 1

        real_lap = F.conv2d(x_complex.real, k.repeat(groups_real,1,1,1) if groups_real > 0 else k, padding=1, groups=groups_real)
        imag_lap = F.conv2d(x_complex.imag, k.repeat(groups_imag,1,1,1) if groups_imag > 0 else k, padding=1, groups=groups_imag)
        return torch.complex(real_lap, imag_lap)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        concat_ch = in_channels // 2 + skip_channels
        self.maxwell_solver = MaxwellSolver(concat_ch)
        self.conv_block1 = BasicConvBlock(concat_ch, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        diffY, diffX = skip_connection.size()[2]-x.size()[2], skip_connection.size()[3]-x.size()[3]
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x_cat = torch.cat([skip_connection, x], dim=1)
        es_tuple = self.maxwell_solver(x_cat)
        out = self.conv_block1(x_cat)
        out = self.conv_block2(out)
        return out, es_tuple
    
class RobustMedVFL_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super().__init__()
        self.enc1, self.pool1 = EncoderBlock(n_channels, 64), nn.MaxPool2d(2)
        self.enc2, self.pool2 = EncoderBlock(64, 128), nn.MaxPool2d(2)
        self.enc3, self.pool3 = EncoderBlock(128, 256), nn.MaxPool2d(2)
        self.enc4, self.pool4 = EncoderBlock(256, 512), nn.MaxPool2d(2)
        self.bottleneck = EncoderBlock(512, 1024)
        self.dec1 = DecoderBlock(1024, 512, 512)
        self.dec2 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec4 = DecoderBlock(128, 64, 64)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        e1=self.enc1(x); p1=self.pool1(e1); e2=self.enc2(p1); p2=self.pool2(e2)
        e3=self.enc3(p2); p3=self.pool3(e3); e4=self.enc4(p3); p4=self.pool4(e4)
        b=self.bottleneck(p4)
        d1,es1=self.dec1(b,e4); d2,es2=self.dec2(d1,e3)
        d3,es3=self.dec3(d2,e2); d4,es4=self.dec4(d3,e1)
        return self.out_conv(d4), (es1, es2, es3, es4)
    
# --- Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-6):
        super().__init__()
        self.num_classes, self.smooth = num_classes, smooth
        
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        loss = 0
        for i in range(self.num_classes):
            inp_c = inputs[:,i,:,:].contiguous().view(-1)
            tgt_c = (targets==i).float().contiguous().view(-1)
            inter = (inp_c * tgt_c).sum()
            dice = (2.*inter+self.smooth)/(inp_c.sum()+tgt_c.sum()+self.smooth)
            loss += (1-dice)
        return loss / self.num_classes

class PhysicsLoss(nn.Module):
    def __init__(self, in_channels_solver):
        super().__init__()
        self.ms = MaxwellSolver(in_channels_solver)
        
    def forward(self, b1, eps, sig):
        b,e,s = b1.to(DEVICE), eps.to(DEVICE), sig.to(DEVICE)
        return torch.mean(self.ms.compute_helmholtz_residual(b,e,s))

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        dy = torch.abs(x[:,:,1:,:]-x[:,:,:-1,:])
        dx = torch.abs(x[:,:,:,1:]-x[:,:,:,:-1])
        return torch.mean(dy) + torch.mean(dx)

class CombinedLoss(nn.Module):
    def __init__(self, wc=.5, wd=.5, wp=.1, ws=.01, in_channels_maxwell=1024, num_classes=4):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.physics_loss = PhysicsLoss(in_channels_solver=in_channels_maxwell)
        self.smoothness_loss = SmoothnessLoss()
        self.wc = wc; self.wd = wd; self.wp = wp; self.ws = ws

    def forward(self, logits, targets, b1, all_es, feat_sm=None):
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        loss_physics = self.physics_loss(b1, all_es[:,:,0], all_es[:,:,1])
        
        total_loss = self.wc * loss_ce + self.wd * loss_dice + self.wp * loss_physics
        
        if feat_sm is not None:
            loss_smoothness = self.smoothness_loss(feat_sm)
            total_loss += self.ws * loss_smoothness
        
        return total_loss


# ===========================================
# DATA-MODEL INTEGRATION FUNCTIONS
# ===========================================

def create_unified_data_loader(
    acdc_data_dir: str = None,
    brats_data_dir: str = None,
    dataset_type: str = "combined",  # "acdc", "brats2020", "combined"
    batch_size: int = 8,
    shuffle: bool = True,
    apply_augmentation: bool = True,
    preprocessor_config: dict = None,
    augmentation_config: dict = None,
    **kwargs
) -> DataLoader:
    """
    Create unified DataLoader for medical datasets.
    
    Args:
        acdc_data_dir: Path to ACDC dataset
        brats_data_dir: Path to BraTS2020 dataset  
        dataset_type: Type of dataset to create ("acdc", "brats2020", "combined")
        batch_size: Batch size
        shuffle: Whether to shuffle data
        apply_augmentation: Whether to apply data augmentation
        preprocessor_config: Preprocessor configuration
        augmentation_config: Augmentation configuration
        **kwargs: Additional arguments
        
    Returns:
        DataLoader instance
    """
    if not UNIFIED_DATA_AVAILABLE:
        raise ImportError("Unified data system not available. Please check imports.")
    
    # Set default paths if not provided
    if acdc_data_dir is None:
        acdc_data_dir = "data/raw/ACDC/database/training"
    if brats_data_dir is None:
        brats_data_dir = "data/raw/BraTS2020_training_data/content/data"
    
    # Create unified loader
    loader = UnifiedFederatedLoader(
        acdc_data_dir=acdc_data_dir,
        brats_data_dir=brats_data_dir,
        preprocessor_config=preprocessor_config,
        augmentation_config=augmentation_config
    )
    
    # Create appropriate DataLoader based on dataset type
    if dataset_type == "acdc":
        return loader.create_single_dataset_loader(
            dataset_type="acdc",
            batch_size=batch_size,
            shuffle=shuffle,
            apply_augmentation=apply_augmentation,
            **kwargs
        )
    elif dataset_type == "brats2020":
        return loader.create_single_dataset_loader(
            dataset_type="brats2020", 
            batch_size=batch_size,
            shuffle=shuffle,
            apply_augmentation=apply_augmentation,
            **kwargs
        )
    elif dataset_type == "combined":
        return loader.create_combined_loader(
            datasets=["acdc", "brats2020"],
            batch_size=batch_size,
            shuffle=shuffle,
            apply_augmentation=apply_augmentation,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def setup_model_and_data(
    model_config: dict = None,
    data_config: dict = None,
    training_config: dict = None
) -> tuple:
    """
    Setup model and data for training.
    
    Args:
        model_config: Model configuration
        data_config: Data configuration  
        training_config: Training configuration
        
    Returns:
        Tuple of (model, train_loader, criterion, optimizer)
    """
    # Default configurations
    model_config = model_config or {}
    data_config = data_config or {}
    training_config = training_config or {}
    
    # Create model
    model = RobustMedVFL_UNet(
        n_channels=model_config.get('n_channels', 1),
        n_classes=model_config.get('n_classes', 4)
    ).to(DEVICE)
    
    # Create data loader
    train_loader = create_unified_data_loader(
        acdc_data_dir=data_config.get('acdc_data_dir'),
        brats_data_dir=data_config.get('brats_data_dir'),
        dataset_type=data_config.get('dataset_type', 'combined'),
        batch_size=data_config.get('batch_size', 8),
        shuffle=data_config.get('shuffle', True),
        apply_augmentation=data_config.get('apply_augmentation', True),
        preprocessor_config=data_config.get('preprocessor_config'),
        augmentation_config=data_config.get('augmentation_config')
    )
    
    # Create loss function
    criterion = CombinedLoss(
        wc=training_config.get('weight_ce', 0.5),
        wd=training_config.get('weight_dice', 0.5),
        wp=training_config.get('weight_physics', 0.1),
        ws=training_config.get('weight_smoothness', 0.01),
        num_classes=model_config.get('n_classes', 4)
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get('learning_rate', 1e-4),
        weight_decay=training_config.get('weight_decay', 1e-5)
    )
    
    return model, train_loader, criterion, optimizer


def train_single_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device = DEVICE,
    apply_quantum_noise: bool = True,
    noise_factor: float = 0.01
) -> dict:
    """
    Train model for a single epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        apply_quantum_noise: Whether to apply quantum noise injection
        noise_factor: Noise factor for quantum noise
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        # Move data to device
        images = images.to(device).float()
        masks = masks.to(device).long()
        
        # Apply quantum noise injection if enabled
        if apply_quantum_noise:
            images = quantum_noise_injection(images, T=noise_factor)
        
        # Forward pass
        optimizer.zero_grad()
        outputs, b1_features = model(images)
        
        # Compute loss (simplified - using only CE loss for now)
        if isinstance(criterion, CombinedLoss):
            # For CombinedLoss, we need additional physics terms
            # Simplified version using only CE and Dice
            loss_ce = nn.CrossEntropyLoss()(outputs, masks)
            loss_dice = DiceLoss()(outputs, masks)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
        else:
            loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'train_loss': avg_loss,
        'num_batches': num_batches,
        'total_samples': num_batches * dataloader.batch_size
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = DEVICE
) -> dict:
    """
    Evaluate model on validation/test data.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_pixels = 0
    dice_scores = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device).float()
            masks = masks.to(device).long()
            
            # Forward pass
            outputs, _ = model(images)
            
            # Compute loss
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Compute accuracy
            predicted = torch.argmax(outputs, dim=1)
            correct_predictions += (predicted == masks).sum().item()
            total_pixels += masks.numel()
            
            # Compute Dice score for each sample
            for i in range(outputs.size(0)):
                dice = compute_dice_score(
                    predicted[i:i+1].unsqueeze(1), 
                    masks[i:i+1].unsqueeze(1),
                    num_classes=4
                )
                dice_scores.append(dice)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_pixels
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    
    return {
        'eval_loss': avg_loss,
        'accuracy': accuracy,
        'dice_score': avg_dice,
        'num_samples': len(dataloader.dataset)
    }


def compute_dice_score(predicted: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> float:
    """
    Compute Dice score for segmentation.
    
    Args:
        predicted: Predicted segmentation
        target: Ground truth segmentation
        num_classes: Number of classes
        
    Returns:
        Average Dice score across all classes
    """
    dice_scores = []
    
    for class_idx in range(num_classes):
        pred_class = (predicted == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / union
            dice_scores.append(dice.item())
    
    return np.mean(dice_scores) if dice_scores else 0.0


def create_federated_client_data(
    client_id: int,
    num_clients: int = 3,
    acdc_data_dir: str = None,
    brats_data_dir: str = None,
    partition_strategy: str = "iid",
    batch_size: int = 8,
    **kwargs
) -> DataLoader:
    """
    Create federated data loader for a specific client.
    
    Args:
        client_id: Client ID (0-based)
        num_clients: Total number of clients
        acdc_data_dir: Path to ACDC dataset
        brats_data_dir: Path to BraTS2020 dataset
        partition_strategy: Partitioning strategy ("iid" or "non_iid")
        batch_size: Batch size
        **kwargs: Additional arguments
        
    Returns:
        DataLoader for the specific client
    """
    if not UNIFIED_DATA_AVAILABLE:
        raise ImportError("Unified data system not available for federated learning.")
    
    # Create unified loader
    loader = UnifiedFederatedLoader(
        acdc_data_dir=acdc_data_dir,
        brats_data_dir=brats_data_dir
    )
    
    # Create federated loaders
    federated_loaders = loader.create_federated_loaders(
        num_clients=num_clients,
        datasets=["acdc", "brats2020"] if acdc_data_dir and brats_data_dir else None,
        partition_strategy=partition_strategy,
        batch_size=batch_size,
        **kwargs
    )
    
    if client_id >= len(federated_loaders):
        raise ValueError(f"Client ID {client_id} exceeds available clients {len(federated_loaders)}")
    
    return federated_loaders[client_id]


# ===========================================
# TRAINING PIPELINE EXAMPLE
# ===========================================

def run_training_example():
    """
    Example of how to use the integrated data-model system.
    """
    print("🚀 Starting Medical Federated Learning Training Example")
    
    # Configuration
    model_config = {
        'n_channels': 1,
        'n_classes': 4
    }
    
    data_config = {
        'dataset_type': 'combined',  # Use both ACDC and BraTS2020
        'batch_size': 4,
        'apply_augmentation': True,
        'preprocessor_config': {
            'target_size': (256, 256),
            'normalize': True
        },
        'augmentation_config': {
            'rotation_range': 15.0,
            'horizontal_flip': True
        }
    }
    
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 2
    }
    
    try:
        # Setup model and data
        model, train_loader, criterion, optimizer = setup_model_and_data(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"✅ Data loader created with {len(train_loader)} batches")
        
        # Training loop
        for epoch in range(training_config['num_epochs']):
            print(f"\n📈 Epoch {epoch + 1}/{training_config['num_epochs']}")
            
            # Train
            train_metrics = train_single_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                apply_quantum_noise=True
            )
            
            print(f"   Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"   Samples: {train_metrics['total_samples']}")
        
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run example when script is executed directly
    run_training_example()