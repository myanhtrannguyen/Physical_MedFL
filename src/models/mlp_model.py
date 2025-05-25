import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import h5py
import glob
import nibabel as nib  # Thêm thư viện để đọc file NIfTI (.nii)
from skimage.transform import resize
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.utils.class_weight import compute_class_weight

# Import the universal data loader
from universal_data_loader import UniversalDataLoader, create_dataloader

# --- Configuration ---
NUM_EPOCHS_CENTRALIZED = 50 
NUM_CLASSES = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
BATCH_SIZE = 8  # Reduced for stability with more data 

# --- File type constants ---
FILE_TYPE_NIFTI = 'nifti'
FILE_TYPE_H5 = 'h5'

# --- Data Augmentation ---
class MedicalDataAugmentation:
    def __init__(self, rotation_degrees=15, flip_prob=0.5, brightness_factor=0.2, contrast_factor=0.2):
        self.rotation_degrees = rotation_degrees
        self.flip_prob = flip_prob
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        
    def __call__(self, image, mask=None):
        # Convert to tensor if numpy
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
            
        # Random horizontal flip
        if torch.rand(1) < self.flip_prob:
            image = torch.flip(image, [-1])
            if mask is not None:
                mask = torch.flip(mask, [-1])
                
        # Random rotation
        if self.rotation_degrees > 0:
            angle = torch.randint(-self.rotation_degrees, self.rotation_degrees + 1, (1,)).item()
            # For medical images, we need to be careful with rotations
            # Apply only small rotations to preserve anatomical structure
            if abs(angle) <= 10:  # Limit rotation to ±10 degrees for medical data
                image = TF.rotate(image, angle, fill=[0.0])
                if mask is not None:
                    mask = TF.rotate(mask, angle, fill=[0.0])
                    
        # Brightness and contrast adjustment (only for image, not mask)
        if torch.rand(1) < 0.5:
            brightness = 1 + (torch.rand(1) - 0.5) * self.brightness_factor
            image = image * brightness
            
        if torch.rand(1) < 0.5:
            contrast = 1 + (torch.rand(1) - 0.5) * self.contrast_factor
            mean = image.mean()
            image = (image - mean) * contrast + mean
            
        # Clamp values to valid range
        image = torch.clamp(image, 0, 1)
        
        return image, mask

def augment_dataset(images, masks, augmentation_factor=3):
    """
    Tăng cường dữ liệu bằng cách tạo ra nhiều phiên bản augmented từ dữ liệu gốc
    """
    augmenter = MedicalDataAugmentation()
    augmented_images = []
    augmented_masks = []
    
    # Giữ lại dữ liệu gốc
    augmented_images.extend(images)
    augmented_masks.extend(masks)
    
    # Tạo augmented data
    for factor in range(augmentation_factor):
        for i in range(len(images)):
            img, mask = augmenter(images[i], masks[i])
            augmented_images.append(img.numpy() if isinstance(img, torch.Tensor) else img)
            augmented_masks.append(mask.numpy() if isinstance(mask, torch.Tensor) else mask)
    
    return np.array(augmented_images), np.array(augmented_masks) 

# --- Standard Convolutional Block ---
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

# # --- Auxiliary Model Components ---
# class ePUREPlaceholder(nn.Module):
#     def __init__(self): super().__init__()
#     def forward(self, x): return torch.zeros_like(x)

# def adaptive_spline_smoothing_placeholder(x, noise_profile): return x

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
        # Ensure input is in NCHW format (B, C, H, W) as required by PyTorch convolutions
        if x.dim() == 4 and x.shape[3] == 1:  # NHWC format [B, H, W, C]
            x = x.permute(0, 3, 1, 2)  # Convert to NCHW [B, C, H, W]
            
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
    # Ensure input is in NCHW format
    if x.dim() == 4 and x.shape[3] == 1:  # NHWC format [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # Convert to NCHW [B, C, H, W]
    
    # Ensure noise_profile is in NCHW format
    if noise_profile.dim() == 4 and noise_profile.shape[3] == 1:  # NHWC format [B, H, W, C]
        noise_profile = noise_profile.permute(0, 3, 1, 2)  # Convert to NCHW [B, C, H, W]
    
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

def quantum_noise_injection(features):
    features_float = features.float()

    if features_float.dim() < 4 or features_float.size(2) < 2 or features_float.size(3) < 2:
        print("Warning: Features too small for quantum noise injection.")
        return features_float # Return original features as float

    try:
        # Ensure tensors are on the correct device
        device = features_float.device
        rotated_features = [
            features_float,
            torch.rot90(features_float, k=1, dims=[-2, -1]),
            torch.rot90(features_float, k=2, dims=[-2, -1])
        ]
        pauli_effect = torch.mean(torch.stack(rotated_features, dim=0), dim=0)
        noise = 0.1 * pauli_effect * torch.randn_like(features_float, device=device)
        return features_float + noise
    except RuntimeError as e:
        print(f"Quantum noise injection failed: {e}. Returning original features.")
        # Return original features as float if error occurs
        return features_float
    
# --- Model Components (U-Net based) ---
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)
        self.noise_estimator = ePURE(in_channels=in_channels)

    def forward(self, x):
        # Ensure input is in NCHW format (B, C, H, W) as required by PyTorch convolutions
        if x.dim() == 4 and x.shape[3] == 1:  # NHWC format [B, H, W, C]
            x = x.permute(0, 3, 1, 2)  # Convert to NCHW [B, C, H, W]
            
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
        # Ensure input is in NCHW format (B, C, H, W) as required by PyTorch convolutions
        if x.dim() == 4 and x.shape[3] == 1:  # NHWC format [B, H, W, C]
            x = x.permute(0, 3, 1, 2)  # Convert to NCHW [B, C, H, W]
            
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
        # Ensure input is in NCHW format (B, C, H, W) as required by PyTorch convolutions
        if x.dim() == 4 and x.shape[3] == 1:  # NHWC format [B, H, W, C]
            x = x.permute(0, 3, 1, 2)  # Convert to NCHW [B, C, H, W]
            
        if skip_connection.dim() == 4 and skip_connection.shape[3] == 1:  # NHWC format [B, H, W, C]
            skip_connection = skip_connection.permute(0, 3, 1, 2)  # Convert to NCHW [B, C, H, W]
            
        x = self.up(x)
        diffY, diffX = skip_connection.size()[2]-x.size()[2], skip_connection.size()[3]-x.size()[3]
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x_cat = torch.cat([skip_connection, x], dim=1)
        es_tuple = self.maxwell_solver(x_cat)
        out = self.conv_block1(x_cat)
        out = self.conv_block2(out)
        return out, es_tuple
    
class RobustMedVFL_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Encoder path
        self.enc1, self.pool1 = EncoderBlock(n_channels, 64), nn.MaxPool2d(2)
        self.enc2, self.pool2 = EncoderBlock(64, 128), nn.MaxPool2d(2)
        self.enc3, self.pool3 = EncoderBlock(128, 256), nn.MaxPool2d(2)
        self.enc4, self.pool4 = EncoderBlock(256, 512), nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = EncoderBlock(512, 1024)
        
        # Add dropout layers for regularization
        self.dropout1 = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout3 = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Decoder path
        self.dec1 = DecoderBlock(1024, 512, 512)
        self.dec2 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec4 = DecoderBlock(128, 64, 64)
        
        # Output conv
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Permute input from [B, H, W, C] to [B, C, H, W]
        if x.dim() == 4 and x.shape[3] == 1: # Assuming 1 channel, last dimension
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 4 and x.shape[1] == 1: # Already [B, C, H, W]
            pass # No permutation needed
        else:
            # Handle other cases or raise an error if the format is unexpected
            # For now, let's assume if it's not [B,H,W,C] with C=1, it might be already correct or needs specific handling
            pass

        # Encoder path with dropout for regularization
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Apply dropout before bottleneck for regularization
        p4 = self.dropout1(p4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Apply dropout after bottleneck
        b = self.dropout2(b)
        
        # Decoder path
        d1, es1 = self.dec1(b, e4)
        d2, es2 = self.dec2(d1, e3)
        
        # Apply dropout in middle of decoder path
        d2 = self.dropout3(d2)
        
        d3, es3 = self.dec3(d2, e2)
        d4, es4 = self.dec4(d3, e1)
        
        # Output layer (no dropout here to preserve spatial details)
        return self.out_conv(d4), (es1, es2, es3, es4)
    
# --- Loss Functions ---
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
            
            # Apply class weight
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

class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        loss = 0
        for i in range(self.num_classes):
            inp_c = inputs[:, i, :, :].contiguous().view(-1)
            tgt_c = (targets == i).float().contiguous().view(-1)
            inter = (inp_c * tgt_c).sum()
            dice = (2. * inter + self.smooth) / (inp_c.sum() + tgt_c.sum() + self.smooth)
            loss += (1 - dice)
        return loss / self.num_classes

def compute_class_weights(masks, num_classes=4):
    """
    Tính toán class weights để xử lý class imbalance
    """
    # Flatten all masks to get class distribution
    all_labels = []
    for mask in masks:
        all_labels.extend(mask.flatten())
    
    # Compute class weights
    unique_classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
    
    print(f"Computed class weights: {class_weights}")
    return class_weights

class PhysicsLoss(nn.Module):
    def __init__(self, in_channels_solver):
        super().__init__(); self.ms = MaxwellSolver(in_channels_solver)
    def forward(self, b1, eps, sig):
        b,e,s = b1.to(DEVICE), eps.to(DEVICE), sig.to(DEVICE)
        return torch.mean(self.ms.compute_helmholtz_residual(b,e,s))

class SmoothnessLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        dy = torch.abs(x[:,:,1:,:]-x[:,:,:-1,:]); dx = torch.abs(x[:,:,:,1:]-x[:,:,:,:-1])
        return torch.mean(dy) + torch.mean(dx)

class CombinedLoss(nn.Module):
    def __init__(self, wc=0.3, wd=0.4, wf=0.3, wp=0.1, ws=0.01, in_channels_maxwell=1024, num_classes=4, class_weights=None):
        super().__init__()
        # Use weighted cross-entropy if class weights are provided
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None)
        self.dl = WeightedDiceLoss(num_classes=num_classes, class_weights=class_weights)
        self.fl = FocalLoss(class_weights=class_weights, num_classes=num_classes)
        self.pl = PhysicsLoss(in_channels_solver=in_channels_maxwell)
        self.sl = SmoothnessLoss()
        self.wc, self.wd, self.wf, self.wp, self.ws = wc, wd, wf, wp, ws
        
        # Move class weights to appropriate device when needed
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, targets, b1, all_es, feat_sm=None):
        # Move class weights to device if needed
        if self.class_weights is not None and self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(logits.device)
            
        lce = self.ce(logits, targets.long())
        ldc = self.dl(logits, targets.long())
        lfl = self.fl(logits, targets.long())
        
        # Combine cross-entropy, dice, and focal losses
        loss = self.wc * lce + self.wd * ldc + self.wf * lfl
        
        lphy = torch.tensor(0., device=logits.device)
        if b1 is not None and all_es and len(all_es) > 0:
            e1, s1 = all_es[0] 
            lphy = self.pl(b1, e1, s1)
            loss += self.wp * lphy
            
        lsm = torch.tensor(0., device=logits.device)
        if feat_sm is not None:
            lsm = self.sl(feat_sm)
            loss += self.ws * lsm
            
        return loss
    
# --- Metrics ---
def evaluate_metrics(model, dataloader, device, num_classes=4):
    if dataloader is None:
        print("Warning: dataloader is None in evaluate_metrics")
        # Return empty metrics
        metrics = {
            'dice_scores': [0.0] * num_classes,
            'iou': [0.0] * num_classes, 
            'precision': [0.0] * num_classes,
            'recall': [0.0] * num_classes,
            'f1_score': [0.0] * num_classes
        }
        return metrics
        
    model.eval()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    dice_s = [0.0] * num_classes
    iou_s = [0.0] * num_classes
    batches = 0

    with torch.no_grad():
        for imgs,tgts in dataloader:
            imgs,tgts = imgs.to(device),tgts.to(device)
            if imgs.size(0) == 0: continue
            logits,_ = model(imgs)
            preds = torch.argmax(F.softmax(logits,dim=1),dim=1); batches+=1
            for c in range(num_classes):
                pc_f,tc_f=(preds==c).float().view(-1),(tgts==c).float().view(-1); inter=(pc_f*tc_f).sum()
                dice_s[c]+=((2.*inter+1e-6)/(pc_f.sum()+tc_f.sum()+1e-6)).item()
                iou_s[c]+=((inter+1e-6)/(pc_f.sum()+tc_f.sum()-inter+1e-6)).item()
                tp[c]+=inter.item(); fp[c]+=(pc_f.sum()-inter).item(); fn[c]+=(tc_f.sum()-inter).item()
    metrics={'dice_scores':[],'iou':[],'precision':[],'recall':[],'f1_score':[]}
    if batches>0:
        for c in range(num_classes):
            metrics['dice_scores'].append(dice_s[c]/batches); metrics['iou'].append(iou_s[c]/batches)
            prec,rec = tp[c]/(tp[c]+fp[c]+1e-6), tp[c]/(tp[c]+fn[c]+1e-6)
            metrics['precision'].append(prec); metrics['recall'].append(rec)
            metrics['f1_score'].append(2*prec*rec/(prec+rec+1e-6) if (prec+rec > 0) else 0.0)
    else: 
        for _ in range(num_classes): [metrics[key].append(0.0) for key in metrics]
    return metrics

# --- Main Execution (Centralized Training) ---
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    
    # --- Load Data using Universal Data Loader ---
    # Initialize universal data loader
    data_loader = UniversalDataLoader(
        target_size=(IMG_SIZE, IMG_SIZE), 
        num_classes=NUM_CLASSES,
        normalize=True
    )
    
    # Check for both preprocessed H5 data and raw NIfTI data
    base_data_path_h5 = '/Users/alvinluong/Documents/ACDC_preprocessed'
    base_data_path_nifti = '/Users/alvinluong/Documents/Federated_Learning/ACDC/database'
    
    # Initialize variables
    X_train_tensor = None
    y_train_tensor = None
    X_val_tensor = None
    y_val_tensor = None
    X_test_tensor = None
    y_test_tensor = None
    test_dataset = None
    test_dataloader = None
    
    # Use NIfTI data if available, otherwise fallback to preprocessed H5 or dummy data
    if os.path.exists(base_data_path_nifti) and os.listdir(base_data_path_nifti):
        print(f"Using NIfTI data from {base_data_path_nifti}")
        train_dir = os.path.join(base_data_path_nifti, 'training')
        test_dir = os.path.join(base_data_path_nifti, 'testing')
        
        # Load training data
        all_train_images_np, all_train_masks_np = data_loader.load_data(
            train_dir, max_samples=600
        )
        
        # Load test data
        all_test_images_np, all_test_masks_np = data_loader.load_data(
            test_dir, max_samples=200
        )
        
        if all_train_images_np.size == 0:
            raise ValueError("Training data is empty after loading. Check data path and content.")
        
        # Compute class weights for handling class imbalance
        if all_train_masks_np is not None:
            class_weights = compute_class_weights(all_train_masks_np, NUM_CLASSES)
        else:
            class_weights = None
        
        if all_test_images_np.size > 0:
            X_test_tensor = torch.tensor(all_test_images_np).permute(0, 3, 1, 2).float()
            if all_test_masks_np is not None:
                y_test_tensor = torch.tensor(all_test_masks_np).long()
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
        
        # Split train/validation
        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            all_train_images_np, all_train_masks_np, test_size=0.2, random_state=42
        )
        
        X_train_tensor = torch.tensor(X_train_np).permute(0, 3, 1, 2).float()
        y_train_tensor = torch.tensor(y_train_np).long()
        X_val_tensor = torch.tensor(X_val_np).permute(0, 3, 1, 2).float()
        y_val_tensor = torch.tensor(y_val_np).long()
        
    elif os.path.exists(base_data_path_h5) and os.listdir(base_data_path_h5):
        print(f"Using preprocessed H5 data from {base_data_path_h5}")
        train_dir = os.path.join(base_data_path_h5, 'ACDC_training_slices')
        test_dir = os.path.join(base_data_path_h5, 'ACDC_testing_volumes')
        
        # Load training data
        all_train_images_np, all_train_masks_np = data_loader.load_data(
            train_dir, max_samples=600
        )
        
        # Load test data  
        all_test_images_np, all_test_masks_np = data_loader.load_data(
            test_dir, max_samples=200
        )
        
        if all_train_images_np.size == 0:
            raise ValueError("Training data is empty after loading. Check data path and content.")
        
        # Compute class weights for handling class imbalance
        if all_train_masks_np is not None:
            class_weights = compute_class_weights(all_train_masks_np, NUM_CLASSES)
        else:
            class_weights = None
        
        if all_test_images_np.size > 0:
            X_test_tensor = torch.tensor(all_test_images_np).permute(0, 3, 1, 2).float()
            if all_test_masks_np is not None:
                y_test_tensor = torch.tensor(all_test_masks_np).long()
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
        
        # Split train/validation
        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            all_train_images_np, all_train_masks_np, test_size=0.2, random_state=42
        )
        
        X_train_tensor = torch.tensor(X_train_np).permute(0, 3, 1, 2).float()
        y_train_tensor = torch.tensor(y_train_np).long()
        X_val_tensor = torch.tensor(X_val_np).permute(0, 3, 1, 2).float()
        y_val_tensor = torch.tensor(y_val_np).long()
        
    else:
        print(f"No data found. Using DUMMY data.")
        # Create dummy data for testing
        X_train_tensor = torch.randn(100, 1, IMG_SIZE, IMG_SIZE)
        y_train_tensor = torch.randint(0, NUM_CLASSES, (100, IMG_SIZE, IMG_SIZE))
        X_val_tensor = torch.randn(20, 1, IMG_SIZE, IMG_SIZE)
        y_val_tensor = torch.randint(0, NUM_CLASSES, (20, IMG_SIZE, IMG_SIZE))
        
        # Create dummy test data as well
        X_test_tensor = torch.randn(10, 1, IMG_SIZE, IMG_SIZE)
        y_test_tensor = torch.randint(0, NUM_CLASSES, (10, IMG_SIZE, IMG_SIZE))
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        class_weights = None

    if X_train_tensor is None or len(X_train_tensor) == 0: 
        raise ValueError("No training samples after split.")
    if X_val_tensor is None or len(X_val_tensor) == 0: 
        print("Warning: Validation set is empty after split.")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print("Data loaded and prepared for centralized training.")

    # --- Initialize Model, Criterion, Optimizer ---
    model = RobustMedVFL_UNet(n_channels=1, n_classes=NUM_CLASSES).to(DEVICE)
    criterion = CombinedLoss(num_classes=NUM_CLASSES, in_channels_maxwell=1024, class_weights=class_weights).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) # Tùy chọn

    # --- Centralized Training Loop ---
    best_val_metric = 0.0 # Hoặc float('inf') nếu loss là metric chính

    for epoch in range(NUM_EPOCHS_CENTRALIZED):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS_CENTRALIZED} ---")
        
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        for images, targets in train_dataloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            images_noisy = quantum_noise_injection(images) # Tùy chọn áp dụng noise
            
            optimizer.zero_grad()
            logits, all_eps_sigma_tuples = model(images_noisy)
            b1_map_placeholder = torch.randn_like(images[:, 0:1, ...], device=DEVICE) # Placeholder
            
            loss = criterion(logits, targets, b1_map_placeholder, all_eps_sigma_tuples) #, features_for_smoothness=None)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
        print(f"  Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        try:
            has_val_data = (val_dataloader is not None and 
                          hasattr(val_dataloader, 'dataset') and 
                          val_dataloader.dataset is not None)
            
            if has_val_data:
                print("  Evaluating on validation set...")
                val_metrics = evaluate_metrics(model, val_dataloader, DEVICE, NUM_CLASSES)
                # Sử dụng Dice score của class foreground trung bình làm metric chính để so sánh
                # Lấy các giá trị foreground (class từ 1 trở đi)
                fg_dice = val_metrics['dice_scores'][1:] if NUM_CLASSES > 1 else [val_metrics['dice_scores'][0]]
                fg_iou = val_metrics['iou'][1:] if NUM_CLASSES > 1 else [val_metrics['iou'][0]]
                fg_precision = val_metrics['precision'][1:] if NUM_CLASSES > 1 else [val_metrics['precision'][0]]
                fg_recall = val_metrics['recall'][1:] if NUM_CLASSES > 1 else [val_metrics['recall'][0]]
                fg_f1 = val_metrics['f1_score'][1:] if NUM_CLASSES > 1 else [val_metrics['f1_score'][0]]
                
                avg_fg_dice = np.mean(fg_dice)
                avg_fg_iou = np.mean(fg_iou)
                avg_fg_precision = np.mean(fg_precision)
                avg_fg_recall = np.mean(fg_recall)
                avg_fg_f1 = np.mean(fg_f1)
                
                print(f"  Epoch {epoch+1} - Validation (Avg Foreground): "
                      f"Dice: {avg_fg_dice:.4f}; IoU: {avg_fg_iou:.4f}; "
                      f"Precision: {avg_fg_precision:.4f}; Recall: {avg_fg_recall:.4f}; F1-score: {avg_fg_f1:.4f}")
                for c_idx in range(NUM_CLASSES):
                    print(f"    Class {c_idx}: Dice: {val_metrics['dice_scores'][c_idx]:.4f}; "
                          f"IoU: {val_metrics['iou'][c_idx]:.4f}; "
                          f"Precision: {val_metrics['precision'][c_idx]:.4f}; "
                          f"Recall: {val_metrics['recall'][c_idx]:.4f}; "
                          f"F1-score: {val_metrics['f1_score'][c_idx]:.4f}")

                # Tùy chọn: Lưu model tốt nhất dựa trên val_metric
                if avg_fg_dice > best_val_metric:
                    best_val_metric = avg_fg_dice
                    # torch.save(model.state_dict(), "best_centralized_model.pth")
                    # print(f"    New best model saved with Val Dice: {best_val_metric:.4f}")
                
                # if scheduler: scheduler.step(avg_val_loss_or_metric) # Nếu dùng scheduler
            else:
                print("  Validation dataset is empty or not properly loaded. Skipping validation.")
        except Exception as e:
                print(f"Error during validation: {e}")

    print("\n--- Centralized Training Finished ---")

    # --- Evaluate on Test Set ---
    try:
        # Check if test_dataloader exists and is not None
        has_test_data = False
        try:
            if ('test_dataloader' in locals() and 
                test_dataloader is not None and 
                hasattr(test_dataloader, 'dataset') and 
                test_dataloader.dataset is not None):
                # Try to determine if dataset has data by attempting to iterate
                try:
                    # Try to get the first item to check if dataset has data
                    iterator = iter(test_dataloader)
                    next(iterator)
                    has_test_data = True
                except (StopIteration, TypeError, AttributeError):
                    # If we can't iterate or dataset is empty
                    has_test_data = False
        except Exception:
            has_test_data = False
        
        if has_test_data:
            print("\n--- Evaluating on Test Set ---")
            test_metrics = evaluate_metrics(model, test_dataloader, DEVICE, NUM_CLASSES)

            fg_dice = test_metrics['dice_scores'][1:] if NUM_CLASSES > 1 else [test_metrics['dice_scores'][0]]
            fg_iou = test_metrics['iou'][1:] if NUM_CLASSES > 1 else [test_metrics['iou'][0]]
            fg_precision = test_metrics['precision'][1:] if NUM_CLASSES > 1 else [test_metrics['precision'][0]]
            fg_recall = test_metrics['recall'][1:] if NUM_CLASSES > 1 else [test_metrics['recall'][0]]
            fg_f1 = test_metrics['f1_score'][1:] if NUM_CLASSES > 1 else [test_metrics['f1_score'][0]]

            print(f"  Test (Avg Foreground): "
                f"Dice: {np.mean(fg_dice):.4f}; IoU: {np.mean(fg_iou):.4f}; "
                f"Precision: {np.mean(fg_precision):.4f}; Recall: {np.mean(fg_recall):.4f}; "
                f"F1-score: {np.mean(fg_f1):.4f}")
            
            for c_idx in range(NUM_CLASSES):
                print(f"    Class {c_idx}: "
                    f"Dice: {test_metrics['dice_scores'][c_idx]:.4f}; "
                    f"IoU: {test_metrics['iou'][c_idx]:.4f}; "
                    f"Precision: {test_metrics['precision'][c_idx]:.4f}; "
                    f"Recall: {test_metrics['recall'][c_idx]:.4f}; "
                    f"F1-score: {test_metrics['f1_score'][c_idx]:.4f}")
        else:
            print("Test dataset not available or empty.")
    except Exception as e:
        print(f"Error during test evaluation: {e}")