import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Dict, Any, List, Tuple
import torchvision.transforms.functional as TF

# FIXED: Import unified data system with proper error handling
UNIFIED_DATA_AVAILABLE = False
try:
    from src.data.research_loader import create_research_dataloader, create_federated_research_loaders
    UNIFIED_DATA_AVAILABLE = True
except ImportError:
    print("Warning: Unified data system not available - using fallback")
    # Create robust fallback functions
    def create_research_dataloader(*args, **kwargs) -> DataLoader:
        """Fallback research dataloader - creates synthetic ACDC-like data"""
        batch_size = kwargs.get('batch_size', 4)
        num_samples = kwargs.get('num_samples', 32)
        
        # Create synthetic cardiac-like data
        dummy_images = torch.randn(num_samples, 1, 256, 256) * 0.5 + 0.5
        dummy_masks = torch.randint(0, 4, (num_samples, 256, 256))
        
        dataset = TensorDataset(dummy_images, dummy_masks)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def create_federated_research_loaders(*args, **kwargs) -> List[DataLoader]:
        """Fallback federated loaders - creates multiple synthetic data partitions"""
        num_clients = kwargs.get('num_clients', 3)
        return [create_research_dataloader(**kwargs) for _ in range(num_clients)]

# --- Configuration Constants ---
NUM_EPOCHS_CENTRALIZED = 50 
NUM_CLASSES = 4
LEARNING_RATE = 1e-4

# --- Device Configuration ---
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Model Device: CUDA GPU")
else:
    DEVICE = torch.device('cpu')
    print("Model Device: CPU")

IMG_SIZE = 256
BATCH_SIZE = 8

# --- Standard Convolutional Block ---
class BasicConvBlock(nn.Module):
    """Standard convolutional block for U-Net with optional batch normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# --- ePURE Implementation ---
class ePURE(nn.Module):
    """Advanced noise estimation using ePURE method"""
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1)  # Ensure output is 1 channel for noise profile
        )

    def forward(self, x):
        x_float = x.float()

        # Estimate a base noise map from the input features
        noise_map_raw = self.conv(x_float)  # Output is [B, 1, H, W]

        # Simple approach: just output the learned map directly.
        # The adaptive smoothing uses sigmoid, so the network learns to output values
        # that sigmoid can map to appropriate blending weights.
        noise_map = noise_map_raw  # [B, 1, H, W]

        return noise_map  # Noise profile estimate (1 channel)

    def estimate_noise(self, x):
        """Compatibility method for legacy code"""
        return self.forward(x)

# --- Adaptive Spline Smoothing Implementation ---
def adaptive_spline_smoothing(x, noise_profile, kernel_size=5, sigma=1.0):
    """
    Apply adaptive smoothing based on noise_profile
    - x: Input image or feature map [B, C, H, W]
    - noise_profile: Noise map [B, 1, H, W] (values from 0 to 1)
    - kernel_size/sigma: Gaussian smoothing parameters
    """
    # Ensure input is float for convolution
    x_float = x.float()

    # Ensure noise_profile is float and 1 channel
    noise_profile_float = noise_profile.float()
    if noise_profile_float.size(1) != 1:
        print(f"Warning: Noise profile expected 1 channel but got {noise_profile_float.size(1)}. Using first channel.")
        noise_profile_float = noise_profile_float[:, :1, :, :]

    # Step 1: Smooth image with Gaussian blur
    if isinstance(kernel_size, int):
        kernel_size_list = [kernel_size, kernel_size]
    else:
        kernel_size_list = list(kernel_size)

    if isinstance(sigma, (int, float)):
        sigma_list = [float(sigma), float(sigma)]
    else:
        sigma_list = list(sigma)

    # Ensure sigma values are positive to avoid issues
    sigma_list = [max(0.1, s) for s in sigma_list]  # Add small epsilon

    try:
        smoothed = TF.gaussian_blur(x_float, kernel_size=kernel_size_list, sigma=sigma_list)
    except Exception as e:
        print(f"Warning: Gaussian blur failed ({e}), using original image")
        smoothed = x_float

    # Step 2: Normalize noise_profile (sigmoid) and expand for correct number of channels
    # Sigmoid ensures blending weights are between 0 and 1
    # Higher noise_profile value should lead to *more* smoothing.
    blending_weights = torch.sigmoid(noise_profile_float)  # [B, 1, H, W]

    # Expand blending_weights to match the number of channels in x
    blending_weights = blending_weights.repeat(1, x_float.size(1), 1, 1)  # [B, C, H, W]

    # Ensure dimensions match for blending
    assert blending_weights.shape == x_float.shape, f"Blending weights shape {blending_weights.shape} does not match input shape {x_float.shape}"

    # Step 3: Blend original and smoothed image
    # Output = (1 - alpha) * Original + alpha * Smoothed
    # where alpha = blending_weights
    weighted_sum = x_float * (1 - blending_weights) + smoothed * blending_weights

    return weighted_sum

# --- Quantum Noise Injection Implementation ---
def quantum_noise_injection(features, T=1.25, pauli_prob={'X': 0.00096, 'Y': 0.00096, 'Z': 0.00096, 'None': 0.99712}):
    """
    Apply quantum noise based on Pauli Noise Injection mechanism for MRI data.
    
    Args:
        features (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        T (float): Noise factor, typically in range [0.5, 1.5].
        pauli_prob (dict): Probability distribution for Pauli gates (X, Y, Z, None).
    
    Returns:
        torch.Tensor: Output tensor with quantum noise applied.
    """
    # Convert features to float
    features_float = features.float()
    
    # Check tensor dimensions
    if features_float.dim() < 4 or features_float.size(2) < 2 or features_float.size(3) < 2:
        print("Warning: Features too small for quantum noise injection.")
        return features_float

    try:
        # Ensure tensor is on correct device
        device = features_float.device
        
        # Normalize Pauli probabilities with factor T
        scaled_prob = {
            'X': pauli_prob['X'] * T,
            'Y': pauli_prob['Y'] * T,
            'Z': pauli_prob['Z'] * T,
            'None': 1.0 - (pauli_prob['X'] + pauli_prob['Y'] + pauli_prob['Z']) * T
        }
        
        # Create random mask to choose Pauli gate
        batch_size, channels, height, width = features_float.shape
        pauli_choices = ['X', 'Y', 'Z', 'None']
        probabilities = [scaled_prob['X'], scaled_prob['Y'], scaled_prob['Z'], scaled_prob['None']]
        choice_tensor = torch.multinomial(
            torch.tensor(probabilities, device=device),
            batch_size * channels * height * width,
            replacement=True
        ).view(batch_size, channels, height, width)
        
        # Initialize output tensor
        noisy_features = features_float.clone()
        
        # Apply Pauli gates
        for i, pauli in enumerate(pauli_choices):
            mask = (choice_tensor == i)
            if pauli == 'X':
                # Pauli X gate: Flip pixel value (assuming normalized values in [0, 1])
                noisy_features[mask] = 1.0 - noisy_features[mask]
            elif pauli == 'Y':
                # Pauli Y gate: Combine bit flip and add random noise
                noisy_features[mask] = 1.0 - noisy_features[mask] + 0.1 * torch.randn_like(noisy_features[mask], device=device)
            elif pauli == 'Z':
                # Pauli Z gate: Change sign of pixel value
                noisy_features[mask] = -noisy_features[mask]
            # 'None': Keep original value
            
        # Ensure pixel values stay in range [0, 1]
        noisy_features = torch.clamp(noisy_features, 0.0, 1.0)
        
        return noisy_features
    
    except RuntimeError as e:
        print(f"Quantum noise injection failed: {e}. Returning original features.")
        return features_float

# --- Maxwell Solver Implementation ---
class MaxwellSolver(nn.Module):
    """Advanced Maxwell equation solver for physics-informed constraints"""
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

    def solve(self, x):
        """Compatibility method for legacy code"""
        return self.forward(x)

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

# --- U-Net Architecture Components ---
class EncoderBlock(nn.Module):
    """U-Net encoder block with ePURE noise estimation"""
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

class DecoderBlock(nn.Module):
    """U-Net decoder block with Maxwell solver"""
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

# --- Main U-Net Model ---
class RobustMedVFL_UNet(nn.Module):
    """
    Advanced Medical VFL U-Net for cardiac segmentation with physics-informed constraints
    - Proper U-Net architecture with ePURE noise estimation
    - Maxwell solver for physics constraints
    - Quantum noise injection for robustness
    - Optimized for federated learning
    """
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
    """Improved Dice Loss for medical segmentation"""
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
    """Physics-informed loss using Maxwell solver"""
    def __init__(self, in_channels_solver):
        super().__init__()
        self.ms = MaxwellSolver(in_channels_solver)
        
    def forward(self, b1, eps, sig):
        b,e,s = b1.to(DEVICE), eps.to(DEVICE), sig.to(DEVICE)
        return torch.mean(self.ms.compute_helmholtz_residual(b,e,s))

class SmoothnessLoss(nn.Module):
    """Smoothness constraint loss"""
    def __init__(self): 
        super().__init__()
        
    def forward(self, x):
        dy = torch.abs(x[:,:,1:,:]-x[:,:,:-1,:])
        dx = torch.abs(x[:,:,:,1:]-x[:,:,:,:-1])
        return torch.mean(dy) + torch.mean(dx)

class CombinedLoss(nn.Module):
    """Advanced combined loss function for medical segmentation with physics constraints"""
    def __init__(self, wc=.5, wd=.5, wp=.1, ws=.01, in_channels_maxwell=1024, num_classes=4):
        super().__init__()
        # Use class weights for ACDC dataset
        class_weights = torch.tensor([0.25, 1.5, 1.3, 1.4])  # Background, RV, Myocardium, LV
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dl = DiceLoss(num_classes=num_classes)
        self.pl = PhysicsLoss(in_channels_solver=in_channels_maxwell)
        self.sl = SmoothnessLoss()
        self.wc,self.wd,self.wp,self.ws = wc,wd,wp,ws

    def forward(self, logits, targets, b1, all_es, feat_sm=None):
        lce = self.ce(logits,targets.long())
        ldc = self.dl(logits,targets.long())
        loss = self.wc*lce + self.wd*ldc
        
        lphy = torch.tensor(0.,device=logits.device)
        if b1 is not None and all_es and len(all_es)>0:
            # Use eps, sigma from most important decoder layer
            e1,s1 = all_es[0] 
            lphy=self.pl(b1,e1,s1)
            loss+=self.wp*lphy
            
        lsm = torch.tensor(0.,device=logits.device)
        if feat_sm is not None:
            lsm=self.sl(feat_sm)
            loss+=self.ws*lsm
        return loss

    def __call__(self, outputs, targets):
        """Compatibility method for standard loss function interface"""
        if isinstance(outputs, tuple):
            logits, auxiliary_outputs = outputs
            # Create placeholder b1 map
            b1_placeholder = torch.randn_like(logits[:, 0:1, ...], device=logits.device)
            result = self.forward(logits, targets, b1_placeholder, auxiliary_outputs)
        else:
            logits = outputs
            b1_placeholder = torch.randn_like(logits[:, 0:1, ...], device=logits.device)
            result = self.forward(logits, targets, b1_placeholder, [])
        
        return result

# --- Utility Functions ---
def compute_dice_score(predicted: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> float:
    """Compute Dice score between predicted and target segmentations"""
    dice_scores = []
    
    for i in range(num_classes):
        pred_i = (predicted == i).float()
        target_i = (target == i).float()
        
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / union
            dice_scores.append(dice.item())
        else:
            dice_scores.append(1.0)  # Perfect score if both are empty
    
    return sum(dice_scores) / len(dice_scores)

# --- Fallback Data Creation ---
def create_unified_data_loader(
    acdc_data_dir: Optional[str] = None,
    brats_data_dir: Optional[str] = None,
    dataset_type: str = "combined",
    batch_size: int = 8,
    shuffle: bool = True,
    apply_augmentation: bool = False,  # FIXED: Default to False for stability
    **kwargs
) -> DataLoader:
    """Create unified data loader with robust fallback"""
    
    if not UNIFIED_DATA_AVAILABLE:
        print("Warning: Using fallback synthetic data loader")
        return create_research_dataloader(batch_size=batch_size, **kwargs)
    
    # If unified data is available, use it
    try:
        return create_research_dataloader(
            data_dir=acdc_data_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
    except Exception as e:
        print(f"Warning: Unified data loader failed ({e}), using fallback")
        return create_research_dataloader(batch_size=batch_size, **kwargs)

# Export important classes and functions
__all__ = [
    'RobustMedVFL_UNet',
    'CombinedLoss', 
    'DiceLoss',
    'PhysicsLoss',
    'SmoothnessLoss',
    'quantum_noise_injection',
    'adaptive_spline_smoothing',
    'ePURE',
    'MaxwellSolver',
    'compute_dice_score',
    'create_unified_data_loader'
]