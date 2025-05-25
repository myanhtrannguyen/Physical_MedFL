import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import h5py
import glob
import nibabel as nib
from skimage.transform import resize
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.utils.class_weight import compute_class_weight

# Import data handling modules
from ..data.loader import create_dataloader, create_acdc_dataloader, create_dataloader_from_paths
from ..data.dataset import ACDCDataset, InMemoryDataset, compute_class_weights as compute_dataset_class_weights
from ..data.preprocessing import MedicalImagePreprocessor, DataAugmentation

# Import evaluation metrics
from ..utils.metrics import evaluate_metrics, compute_class_weights, print_metrics_summary

# --- Configuration ---
NUM_EPOCHS_CENTRALIZED = 50
NUM_CLASSES = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
BATCH_SIZE = 8

# --- File type constants ---
FILE_TYPE_NIFTI = 'nifti'
FILE_TYPE_H5 = 'h5'

# --- Optimized Data Augmentation ---
class MedicalDataAugmentation:
    def __init__(self, rotation_degrees=10, flip_prob=0.5, brightness_factor=0.15, contrast_factor=0.15):
        # Reduced parameters for medical safety
        self.rotation_degrees = rotation_degrees
        self.flip_prob = flip_prob
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def __call__(self, image, mask=None):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        # Random horizontal flip
        if torch.rand(1) < self.flip_prob:
            image = torch.flip(image, [-1])
            if mask is not None:
                mask = torch.flip(mask, [-1])

        # Limited rotation for medical safety
        if self.rotation_degrees > 0:
            angle = torch.randint(-self.rotation_degrees, self.rotation_degrees + 1, (1,)).item()
            if abs(angle) <= 10:
                image = TF.rotate(image, angle, fill=[0.0])
                if mask is not None:
                    mask = TF.rotate(mask, angle, fill=[0.0])

        # Reduced brightness/contrast adjustment
        if torch.rand(1) < 0.3:  # Reduced probability
            brightness = 1 + (torch.rand(1) - 0.5) * self.brightness_factor
            image = image * brightness

        if torch.rand(1) < 0.3:  # Reduced probability
            contrast = 1 + (torch.rand(1) - 0.5) * self.contrast_factor
            mean = image.mean()
            image = (image - mean) * contrast + mean

        image = torch.clamp(image, 0, 1)
        return image, mask

# --- Optimized Basic Components ---
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# --- Optimized ePURE Implementation ---
class ePURE(nn.Module):
    def __init__(self, in_channels, base_channels=24):  # Reduced from 32
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1)
        )

    def forward(self, x):
        if x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)
        x_float = x.float()
        noise_map_raw = self.conv(x_float)
        return noise_map_raw

# --- Optimized Adaptive Spline Smoothing ---
def adaptive_spline_smoothing(x, noise_profile, kernel_size=3, sigma=0.8):  # Reduced parameters
    if x.dim() == 4 and x.shape[3] == 1:
        x = x.permute(0, 3, 1, 2)
    if noise_profile.dim() == 4 and noise_profile.shape[3] == 1:
        noise_profile = noise_profile.permute(0, 3, 1, 2)

    x_float = x.float()
    noise_profile_float = noise_profile.float()
    
    if noise_profile_float.size(1) != 1:
        noise_profile_float = noise_profile_float[:, :1, :, :]

    # Simplified smoothing
    if isinstance(kernel_size, int):
        kernel_size_tuple = (kernel_size, kernel_size)
    else:
        kernel_size_tuple = kernel_size

    if isinstance(sigma, (int, float)):
        sigma_tuple = (float(sigma), float(sigma))
    else:
        sigma_tuple = sigma

    sigma_tuple = tuple(max(0.1, s) for s in sigma_tuple)
    smoothed = TF.gaussian_blur(x_float, kernel_size=list(kernel_size_tuple), sigma=list(sigma_tuple))
    
    blending_weights = torch.sigmoid(noise_profile_float)
    blending_weights = blending_weights.repeat(1, x_float.size(1), 1, 1)
    
    weighted_sum = x_float * (1 - blending_weights) + smoothed * blending_weights
    return weighted_sum

# --- Optimized Quantum Noise Injection ---
def optimized_quantum_noise_injection(features, noise_factor=0.05, skip=False):  # Reduced default
    if skip or noise_factor <= 0:
        return features.float()
    
    features_float = features.float()
    if features_float.dim() < 4 or features_float.size(2) < 2 or features_float.size(3) < 2:
        return features_float

    try:
        device = features_float.device
        # Simplified quantum noise - single rotation
        rotated = torch.rot90(features_float, k=1, dims=[-2, -1])
        pauli_effect = (features_float + rotated) / 2
        noise = noise_factor * pauli_effect * torch.randn_like(features_float, device=device)
        return features_float + noise
    except RuntimeError as e:
        print(f"Quantum noise injection failed: {e}")
        return features_float

# --- Optimized Encoder Block ---
class OptimizedEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_noise_processing=True):
        super().__init__()
        self.use_noise_processing = use_noise_processing
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)
        
        if use_noise_processing:
            self.noise_estimator = ePURE(in_channels=in_channels)
        else:
            self.noise_estimator = None

    def forward(self, x):
        if x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)

        if self.noise_estimator is not None:
            noise_profile = self.noise_estimator(x)
            x_smoothed = adaptive_spline_smoothing(x, noise_profile)
            x = self.conv_block1(x_smoothed)
        else:
            x = self.conv_block1(x)
        
        x = self.conv_block2(x)
        return x

# --- Optimized Maxwell Solver ---
class MaxwellSolver(nn.Module):
    def __init__(self, in_channels, hidden_dim=24):  # Reduced from 32
        super(MaxwellSolver, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        )
        omega, mu_0, eps_0 = 2 * np.pi * 42.58e6, 4 * np.pi * 1e-7, 8.854187817e-12
        self.k0 = torch.tensor(omega * np.sqrt(mu_0 * eps_0), dtype=torch.float32)

    def forward(self, x):
        if x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)
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
        groups_real = x_complex.real.size(1) if x_complex.real.size(1) > 0 else 1
        groups_imag = x_complex.imag.size(1) if x_complex.imag.size(1) > 0 else 1
        real_lap = F.conv2d(x_complex.real, k.repeat(groups_real,1,1,1) if groups_real > 0 else k, padding=1, groups=groups_real)
        imag_lap = F.conv2d(x_complex.imag, k.repeat(groups_imag,1,1,1) if groups_imag > 0 else k, padding=1, groups=groups_imag)
        return torch.complex(real_lap, imag_lap)

# --- Optimized Decoder Block ---
class OptimizedDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_maxwell=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        concat_ch = in_channels // 2 + skip_channels
        self.use_maxwell = use_maxwell
        
        if use_maxwell:
            self.maxwell_solver = MaxwellSolver(concat_ch)
        else:
            self.maxwell_solver = None
            
        self.conv_block1 = BasicConvBlock(concat_ch, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)

    def forward(self, x, skip_connection):
        if x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)
        if skip_connection.dim() == 4 and skip_connection.shape[3] == 1:
            skip_connection = skip_connection.permute(0, 3, 1, 2)

        x = self.up(x)
        diffY, diffX = skip_connection.size()[2]-x.size()[2], skip_connection.size()[3]-x.size()[3]
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x_cat = torch.cat([skip_connection, x], dim=1)
        
        es_tuple = self.maxwell_solver(x_cat) if self.use_maxwell and self.maxwell_solver is not None else (None, None)
        
        out = self.conv_block1(x_cat)
        out = self.conv_block2(out)
        return out, es_tuple

# --- Optimized Main Model ---
class OptimizedRobustMedVFL_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, dropout_rate=0.1, pruning_config=None):
        super().__init__()
        
        # Default optimized pruning configuration
        if pruning_config is None:
            pruning_config = {
                'noise_processing_levels': [0, 1],  # Only first 2 encoder levels
                'maxwell_solver_levels': [0, 1],    # Only first 2 decoder levels
                'dropout_positions': [0],           # Only one dropout position
                'skip_quantum_noise': False
            }
        
        self.pruning_config = pruning_config
        
        # Optimized encoder path
        self.enc1 = OptimizedEncoderBlock(
            n_channels, 64, 
            use_noise_processing=(0 in pruning_config['noise_processing_levels'])
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = OptimizedEncoderBlock(
            64, 128,
            use_noise_processing=(1 in pruning_config['noise_processing_levels'])
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = OptimizedEncoderBlock(
            128, 256,
            use_noise_processing=(2 in pruning_config['noise_processing_levels'])
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = OptimizedEncoderBlock(
            256, 512,
            use_noise_processing=(3 in pruning_config['noise_processing_levels'])
        )
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck - always keep noise processing
        self.bottleneck = OptimizedEncoderBlock(512, 1024, use_noise_processing=True)
        
        # Selective dropout
        self.dropout = nn.Dropout2d(p=dropout_rate) if 0 in pruning_config['dropout_positions'] else nn.Identity()
        
        # Optimized decoder path
        self.dec1 = OptimizedDecoderBlock(
            1024, 512, 512,
            use_maxwell=(0 in pruning_config['maxwell_solver_levels'])
        )
        self.dec2 = OptimizedDecoderBlock(
            512, 256, 256,
            use_maxwell=(1 in pruning_config['maxwell_solver_levels'])
        )
        self.dec3 = OptimizedDecoderBlock(
            256, 128, 128,
            use_maxwell=(2 in pruning_config['maxwell_solver_levels'])
        )
        self.dec4 = OptimizedDecoderBlock(
            128, 64, 64,
            use_maxwell=(3 in pruning_config['maxwell_solver_levels'])
        )
        
        # Output conv
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Input format handling
        if x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)
        
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        b = self.dropout(b)
        
        # Decoder path
        d1, es1 = self.dec1(b, e4)
        d2, es2 = self.dec2(d1, e3)
        d3, es3 = self.dec3(d2, e2)
        d4, es4 = self.dec4(d3, e1)
        
        # Filter valid es_tuples
        valid_es_tuples = [es for es in [es1, es2, es3, es4] if es[0] is not None]
        
        return self.out_conv(d4), valid_es_tuples

# --- Optimized Loss Functions ---
class WeightedDiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-6, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        loss = 0
        for i in range(self.num_classes):
            inp_c = inputs[:, i, :, :].contiguous().view(-1)
            tgt_c = (targets == i).float().contiguous().view(-1)
            inter = (inp_c * tgt_c).sum()
            dice = (2. * inter + self.smooth) / (inp_c.sum() + tgt_c.sum() + self.smooth)
            
            if self.class_weights is not None:
                weight = self.class_weights[i].to(inputs.device)
                loss += weight * (1 - dice)
            else:
                loss += (1 - dice)
        return loss / self.num_classes

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, class_weights=None, num_classes=4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        device_weights = None
        if self.class_weights is not None:
            device_weights = self.class_weights.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=device_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class PhysicsLoss(nn.Module):
    def __init__(self, in_channels_solver):
        super().__init__()
        self.ms = MaxwellSolver(in_channels_solver)

    def forward(self, b1, eps, sig):
        b, e, s = b1.to(DEVICE), eps.to(DEVICE), sig.to(DEVICE)
        return torch.mean(self.ms.compute_helmholtz_residual(b, e, s))

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        dy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])
        dx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
        return torch.mean(dy) + torch.mean(dx)

# --- Optimized Combined Loss ---
class OptimizedCombinedLoss(nn.Module):
    def __init__(self, wc=0.4, wd=0.4, wf=0.2, wp=0.05, ws=0.01, 
                 num_classes=4, class_weights=None):
        super().__init__()
        # Reduced physics weight since fewer Maxwell solvers
        self.wc, self.wd, self.wf, self.wp, self.ws = wc, wd, wf, wp, ws
        
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        )
        self.dl = WeightedDiceLoss(num_classes=num_classes, class_weights=class_weights)
        self.fl = FocalLoss(class_weights=class_weights, num_classes=num_classes)
        self.pl = PhysicsLoss(in_channels_solver=1024)
        self.sl = SmoothnessLoss()

    def forward(self, logits, targets, b1, all_es, feat_sm=None):
        # Core losses
        lce = self.ce(logits, targets.long())
        ldc = self.dl(logits, targets.long())
        lfl = self.fl(logits, targets.long())
        
        loss = self.wc * lce + self.wd * ldc + self.wf * lfl
        
        # Physics loss - only if valid es_tuples exist
        lphy = torch.tensor(0., device=logits.device)
        if b1 is not None and all_es and len(all_es) > 0:
            e1, s1 = all_es[0]
            if e1 is not None and s1 is not None:
                lphy = self.pl(b1, e1, s1)
                loss += self.wp * lphy
        
        # Smoothness loss
        lsm = torch.tensor(0., device=logits.device)
        if feat_sm is not None:
            lsm = self.sl(feat_sm)
            loss += self.ws * lsm
        
        return loss

# --- Utility Functions ---
# Metrics functions moved to src/utils/metrics.py

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    
    # Initialize medical image preprocessor
    preprocessor = MedicalImagePreprocessor(
        target_size=(IMG_SIZE, IMG_SIZE),
        normalize=True
    )

    # Data paths
    base_data_path_h5 = '/Users/alvinluong/Documents/ACDC_preprocessed'
    base_data_path_nifti = '/Users/alvinluong/Documents/Federated_Learning/ACDC/database'

    # Initialize variables
    X_train_tensor = None
    y_train_tensor = None
    X_val_tensor = None
    y_val_tensor = None
    X_test_tensor = None
    y_test_tensor = None
    test_dataloader = None
    class_weights = None

    # Load data (same as original but with optimized processing)
    if os.path.exists(base_data_path_nifti) and os.listdir(base_data_path_nifti):
        print(f"Using NIfTI data from {base_data_path_nifti}")
        train_dir = os.path.join(base_data_path_nifti, 'training')
        test_dir = os.path.join(base_data_path_nifti, 'testing')

        # Create ACDC dataset for training
        train_dataset = ACDCDataset(
            data_dir=train_dir,
            preprocessor=preprocessor,
            num_classes=NUM_CLASSES
        )
        
        # Create ACDC dataset for testing
        test_dataset = ACDCDataset(
            data_dir=test_dir,
            preprocessor=preprocessor,
            num_classes=NUM_CLASSES
        )
        
        # Extract data from datasets for compatibility with existing code
        all_train_images_np = []
        all_train_masks_np = []
        for i in range(min(600, len(train_dataset))):
            img, mask = train_dataset[i]
            all_train_images_np.append(img.squeeze().numpy())
            all_train_masks_np.append(mask.numpy())
        
        all_test_images_np = []
        all_test_masks_np = []
        for i in range(min(200, len(test_dataset))):
            img, mask = test_dataset[i]
            all_test_images_np.append(img.squeeze().numpy())
            all_test_masks_np.append(mask.numpy())
        
        # Convert to numpy arrays
        all_train_images_np = np.array(all_train_images_np)
        all_train_masks_np = np.array(all_train_masks_np)
        all_test_images_np = np.array(all_test_images_np)
        all_test_masks_np = np.array(all_test_masks_np)

        if all_train_images_np.size == 0:
            raise ValueError("Training data is empty after loading.")

        class_weights = compute_class_weights(all_train_masks_np, NUM_CLASSES) if all_train_masks_np is not None else None

        if all_test_images_np.size > 0:
            X_test_tensor = torch.tensor(all_test_images_np).permute(0, 3, 1, 2).float()
            if all_test_masks_np is not None:
                y_test_tensor = torch.tensor(all_test_masks_np).long()
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            else:
                y_test_tensor = None
                test_dataloader = None

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            all_train_images_np, all_train_masks_np, test_size=0.2, random_state=42
        )

        X_train_tensor = torch.tensor(X_train_np).permute(0, 3, 1, 2).float()
        y_train_tensor = torch.tensor(y_train_np).long()
        X_val_tensor = torch.tensor(X_val_np).permute(0, 3, 1, 2).float()
        y_val_tensor = torch.tensor(y_val_np).long()

    elif os.path.exists(base_data_path_h5) and os.listdir(base_data_path_h5):
        # Similar processing for H5 data
        print(f"Using preprocessed H5 data from {base_data_path_h5}")
        # ... (same logic as NIfTI)
    else:
        print("No data found. Using DUMMY data.")
        X_train_tensor = torch.randn(100, 1, IMG_SIZE, IMG_SIZE)
        y_train_tensor = torch.randint(0, NUM_CLASSES, (100, IMG_SIZE, IMG_SIZE))
        X_val_tensor = torch.randn(20, 1, IMG_SIZE, IMG_SIZE)
        y_val_tensor = torch.randint(0, NUM_CLASSES, (20, IMG_SIZE, IMG_SIZE))
        X_test_tensor = torch.randn(10, 1, IMG_SIZE, IMG_SIZE)
        y_test_tensor = torch.randint(0, NUM_CLASSES, (10, IMG_SIZE, IMG_SIZE))
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        class_weights = None

    # Create datasets and dataloaders
    if X_train_tensor is not None and y_train_tensor is not None:
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    else:
        train_dataset = None
        train_dataloader = None
    
    if X_val_tensor is not None and y_val_tensor is not None:
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    else:
        val_dataset = None
        val_dataloader = None

    train_size = len(train_dataset) if train_dataset is not None else 0
    val_size = len(val_dataset) if val_dataset is not None else 0
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    print("Data loaded and prepared for optimized centralized training.")

    # Initialize optimized model with balanced pruning config
    balanced_config = {
        'noise_processing_levels': [0, 1],
        'maxwell_solver_levels': [0, 1],
        'dropout_positions': [0],
        'skip_quantum_noise': False
    }

    model = OptimizedRobustMedVFL_UNet(
        n_channels=1, 
        n_classes=NUM_CLASSES, 
        pruning_config=balanced_config
    ).to(DEVICE)

    criterion = OptimizedCombinedLoss(
        num_classes=NUM_CLASSES, 
        class_weights=class_weights
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_metric = 0.0
    for epoch in range(NUM_EPOCHS_CENTRALIZED):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS_CENTRALIZED} ---")

        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        if train_dataloader is None:
            print("No training data available. Skipping training.")
            break
            
        for images, targets in train_dataloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            # Apply optimized quantum noise injection
            if not balanced_config['skip_quantum_noise']:
                images_noisy = optimized_quantum_noise_injection(images, noise_factor=0.05)
            else:
                images_noisy = images

            optimizer.zero_grad()
            logits, all_eps_sigma_tuples = model(images_noisy)
            b1_map_placeholder = torch.randn_like(images[:, 0:1, ...], device=DEVICE)
            loss = criterion(logits, targets, b1_map_placeholder, all_eps_sigma_tuples)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
        print(f" Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        try:
            if val_dataloader is not None:
                print(" Evaluating on validation set...")
                val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, NUM_CLASSES)
                
                fg_dice = val_metrics['dice_scores'][1:] if NUM_CLASSES > 1 else [val_metrics['dice_scores'][0]]
                avg_fg_dice = np.mean(fg_dice)
                
                print(f" Epoch {epoch+1} - Validation Avg Foreground Dice: {avg_fg_dice:.4f}")
                
                if avg_fg_dice > best_val_metric:
                    best_val_metric = avg_fg_dice
                    print(f" New best validation Dice: {best_val_metric:.4f}")
                    
                    # Print detailed metrics for best epoch
                    if epoch % 10 == 0 or avg_fg_dice > best_val_metric:
                        class_names = ['Background', 'RV', 'Myocardium', 'LV']
                        print_metrics_summary(val_metrics, class_names)
            else:
                print(" Validation dataset not available.")
        except Exception as e:
            print(f"Error during validation: {e}")

    print("\n--- Optimized Centralized Training Finished ---")

    # Test evaluation
    try:
        if test_dataloader is not None:
            print("\n--- Evaluating on Test Set ---")
            test_metrics = evaluate_metrics(model, test_dataloader, DEVICE, NUM_CLASSES)
            fg_dice = test_metrics['dice_scores'][1:] if NUM_CLASSES > 1 else [test_metrics['dice_scores'][0]]
            print(f" Test Avg Foreground Dice: {np.mean(fg_dice):.4f}")
            
            # Print detailed test metrics
            class_names = ['Background', 'RV', 'Myocardium', 'LV']
            print_metrics_summary(test_metrics, class_names)
        else:
            print("Test dataset not available.")
    except Exception as e:
        print(f"Error during test evaluation: {e}")
