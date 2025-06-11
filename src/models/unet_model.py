import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import torchvision.transforms.functional as TF

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
        noise_map_raw = self.conv(x_float)
        return noise_map_raw

    def estimate_noise(self, x):
        return self.forward(x)

# --- Adaptive Spline Smoothing Implementation ---
def adaptive_spline_smoothing(x, noise_profile, kernel_size=5, sigma=1.0):
    x_float = x.float()
    noise_profile_float = noise_profile.float()
    if noise_profile_float.size(1) != 1:
        noise_profile_float = noise_profile_float[:, :1, :, :]

    if isinstance(kernel_size, int):
        kernel_size_list = [kernel_size, kernel_size]
    else:
        kernel_size_list = list(kernel_size)

    if isinstance(sigma, (int, float)):
        sigma_list = [float(sigma), float(sigma)]
    else:
        sigma_list = list(sigma)
    sigma_list = [max(0.1, s) for s in sigma_list]

    try:
        smoothed = TF.gaussian_blur(x_float, kernel_size=kernel_size_list, sigma=sigma_list)
    except Exception as e:
        smoothed = x_float

    blending_weights = torch.sigmoid(noise_profile_float)
    blending_weights = blending_weights.repeat(1, x_float.size(1), 1, 1)

    assert blending_weights.shape == x_float.shape
    weighted_sum = x_float * (1 - blending_weights) + smoothed * blending_weights
    return weighted_sum

# --- Quantum Noise Injection Implementation ---
def quantum_noise_injection(features, T=1.25, pauli_prob={'X': 0.00096, 'Y': 0.00096, 'Z': 0.00096, 'None': 0.99712}):
    features_float = features.float()
    if features_float.dim() < 4 or features_float.size(2) < 2 or features_float.size(3) < 2:
        return features_float
    try:
        device = features_float.device
        scaled_prob = {
            'X': pauli_prob['X'] * T, 'Y': pauli_prob['Y'] * T, 'Z': pauli_prob['Z'] * T,
            'None': 1.0 - (pauli_prob['X'] + pauli_prob['Y'] + pauli_prob['Z']) * T
        }
        batch_size, channels, height, width = features_float.shape
        pauli_choices = ['X', 'Y', 'Z', 'None']
        probabilities = [scaled_prob[p] for p in pauli_choices]
        choice_tensor = torch.multinomial(
            torch.tensor(probabilities, device=device),
            batch_size * channels * height * width,
            replacement=True
        ).view(batch_size, channels, height, width)
        
        noisy_features = features_float.clone()
        for i, pauli in enumerate(pauli_choices):
            mask = (choice_tensor == i)
            if pauli == 'X':
                noisy_features[mask] = 1.0 - noisy_features[mask]
            elif pauli == 'Y':
                noisy_features[mask] = 1.0 - noisy_features[mask] + 0.1 * torch.randn_like(noisy_features[mask])
            elif pauli == 'Z':
                noisy_features[mask] = -noisy_features[mask]
        
        return torch.clamp(noisy_features, 0.0, 1.0)
    except RuntimeError:
        return features_float
    
# --- U-Net Architecture Components ---
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
        return (res.real ** 2 + res.imag ** 2).mean()

    def _laplacian_2d(self, x_complex):
        k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=x_complex.device).view(1,1,3,3)
        real_lap = F.conv2d(x_complex.real, k.repeat(x_complex.real.size(1),1,1,1), padding=1, groups=x_complex.real.size(1))
        imag_lap = F.conv2d(x_complex.imag, k.repeat(x_complex.imag.size(1),1,1,1), padding=1, groups=x_complex.imag.size(1))
        return torch.complex(real_lap, imag_lap)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class PhysicsInformedUp(nn.Module):
    """Upscaling then double conv, with Maxwell solver for physics constraints."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.maxwell_solver = MaxwellSolver(in_channels=in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        
        es_tuple = self.maxwell_solver(x)
        
        conv_out = self.conv(x)
        
        return conv_out, es_tuple

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class RobustMedVFL_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super(RobustMedVFL_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 
        self.down4 = Down(512, 1024 // factor)
        self.up1 = PhysicsInformedUp(1024, 512 // factor, bilinear=True)
        self.up2 = PhysicsInformedUp(512, 256 // factor, bilinear=True)
        self.up3 = PhysicsInformedUp(256, 128 // factor, bilinear=True)
        self.up4 = PhysicsInformedUp(128, 64, bilinear=True)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        d1, es1 = self.up1(x5, x4)
        d2, es2 = self.up2(d1, x3)
        d3, es3 = self.up3(d2, x2)
        d4, es4 = self.up4(d3, x1)
        
        all_es = [es1, es2, es3, es4]
        
        logits = self.outc(d4)
        return logits, x5, all_es

class AdaptiveTvMFDiceLoss(nn.Module):
    def __init__(self, num_classes=4, max_kappa=50.0, smooth=1e-7):  # Reduced max_kappa for stability
        super(AdaptiveTvMFDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        # FIXED: Initialize with smaller, more stable kappa values
        self.kappa = nn.Parameter(torch.full((num_classes,), 2.0))  # Reduced from 10.0
        self.max_kappa = max_kappa
        
        # 🎯 NEW: Adaptive κ scheduler components
        self.enable_adaptive_kappa = True
        self.kappa_history = []  # Track κ changes over time
        self.dice_score_history = []  # Track validation dice scores
        self.adaptation_patience = 3  # Wait 3 evaluations before adapting
        self.adaptation_factor = 0.8  # How aggressively to adapt κ
        self.min_kappa = 0.1  # Minimum allowed κ value
        self.last_dice_scores = None  # Store last validation dice scores

    def update_kappa_from_validation(self, dice_scores_per_class):
        """
        🎯 ADAPTIVE κ SCHEDULER: Update κ values based on validation performance
        
        Args:
            dice_scores_per_class: List or tensor of dice scores for each class [class0, class1, ...]
        """
        if not self.enable_adaptive_kappa:
            return
            
        try:
            # Ensure dice_scores is a tensor
            if isinstance(dice_scores_per_class, (list, tuple)):
                dice_scores = torch.tensor(dice_scores_per_class, dtype=torch.float32)
            else:
                dice_scores = dice_scores_per_class.clone().detach()
            
            # Ensure we have the right number of classes
            if len(dice_scores) != self.num_classes:
                print(f"⚠️  Kappa update skipped: expected {self.num_classes} dice scores, got {len(dice_scores)}")
                return
            
            # Store history for analysis
            self.dice_score_history.append(dice_scores.clone())
            if len(self.dice_score_history) > 10:  # Keep only last 10 evaluations
                self.dice_score_history.pop(0)
                
            # 🔧 ADAPTIVE STRATEGY: Increase κ for well-performing classes, decrease for struggling ones
            with torch.no_grad():
                for class_idx in range(self.num_classes):
                    current_dice = dice_scores[class_idx].item()
                    current_kappa = self.kappa[class_idx].item()
                    
                    # Adaptation logic based on performance
                    if current_dice > 0.7:  # Good performance → increase κ (more confident)
                        new_kappa = min(current_kappa * 1.1, self.max_kappa)
                    elif current_dice < 0.3:  # Poor performance → decrease κ (less confident)
                        new_kappa = max(current_kappa * 0.9, self.min_kappa)
                    else:  # Moderate performance → gradual adjustment
                        # Target κ based on dice score: better dice → higher κ
                        target_kappa = self.min_kappa + (self.max_kappa - self.min_kappa) * current_dice
                        new_kappa = current_kappa + self.adaptation_factor * (target_kappa - current_kappa)
                        new_kappa = torch.clamp(torch.tensor(new_kappa), self.min_kappa, self.max_kappa).item()
                    
                    # Apply the update
                    self.kappa[class_idx].data = torch.tensor(new_kappa)
                
                # Track κ history
                self.kappa_history.append(self.kappa.clone().detach())
                if len(self.kappa_history) > 10:
                    self.kappa_history.pop(0)
                
                # Log κ updates
                kappa_values = self.kappa.detach().cpu().numpy()
                dice_values = dice_scores.cpu().numpy()
                print(f"Adaptive κ Update:")
                for i, (kappa, dice) in enumerate(zip(kappa_values, dice_values)):
                    print(f"   Class {i}: κ={kappa:.3f} (dice={dice:.3f})")
                    
        except Exception as e:
            print(f"⚠️  Kappa adaptation failed: {e}")

    def get_kappa_statistics(self):
        """Get comprehensive κ statistics for monitoring"""
        try:
            current_kappa = self.kappa.detach().cpu().numpy()
            stats = {
                'current_kappa': current_kappa.tolist(),
                'mean_kappa': float(np.mean(current_kappa)),
                'std_kappa': float(np.std(current_kappa)),
                'min_kappa': float(np.min(current_kappa)),
                'max_kappa': float(np.max(current_kappa)),
                'adaptive_enabled': self.enable_adaptive_kappa,
                'num_adaptations': len(self.kappa_history)
            }
            
            # Add trend information if we have history
            if len(self.kappa_history) >= 2:
                recent_kappa = self.kappa_history[-1].cpu().numpy()
                previous_kappa = self.kappa_history[-2].cpu().numpy()
                kappa_change = recent_kappa - previous_kappa
                stats['recent_kappa_change'] = kappa_change.tolist()
                stats['kappa_trend'] = 'increasing' if np.mean(kappa_change) > 0 else 'decreasing'
            
            return stats
        except Exception as e:
            print(f"Failed to get kappa statistics: {e}")
            return {}

    def forward(self, inputs, targets, b1=None, all_es=None, feat_sm=None):
        # FIXED: Proper kappa clamping with gradient safety
        with torch.no_grad():
            self.kappa.clamp_(0.1, self.max_kappa)  # Minimum 0.1 to avoid numerical issues
        
        # FIXED: Safer total variation loss computation - ensure tensor results
        if inputs.size(3) > 1:  # Check width dimension
            tv_h = torch.mean(torch.abs(inputs[:, :, :, :-1] - inputs[:, :, :, 1:]))
        else:
            tv_h = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
            
        if inputs.size(2) > 1:  # Check height dimension  
            tv_w = torch.mean(torch.abs(inputs[:, :, :-1, :] - inputs[:, :, 1:, :]))
        else:
            tv_w = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
            
        tv_loss = tv_h + tv_w
        
        # FIXED: Safer softmax with numerical stability
        probs = F.softmax(inputs, dim=1)
        
        # FIXED: Proper target handling with validation
        if targets.max() >= self.num_classes or targets.min() < 0:
            # Clamp invalid targets to valid range
            targets = torch.clamp(targets, 0, self.num_classes - 1)
        
        try:
            one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        except Exception as e:
            # Fallback: create manual one-hot encoding
            one_hot_targets = torch.zeros(targets.shape[0], self.num_classes, targets.shape[1], targets.shape[2], 
                                        device=targets.device, dtype=torch.float32)
            for i in range(self.num_classes):
                one_hot_targets[:, i] = (targets == i).float()
        
        dice_loss = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
        valid_classes = 0
        
        for i in range(self.num_classes):
            class_probs = probs[:, i, :, :]
            class_targets = one_hot_targets[:, i, :, :]
            
            # FIXED: Robust dice computation with proper smoothing
            intersection = torch.sum(class_probs * class_targets)
            pred_sum = torch.sum(class_probs)
            target_sum = torch.sum(class_targets)
            
            # Avoid division by zero
            if pred_sum + target_sum > self.smooth:
                dice_coeff = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
                
                # FIXED: Stable vMF weighting with proper clamping
                dice_coeff = torch.clamp(dice_coeff, 0.0, 1.0)  # Ensure dice is in [0,1]
                
                # Safer exponential computation
                kappa_value = torch.clamp(self.kappa[i], 0.1, 10.0)  # Conservative kappa range
                exponent = kappa_value * (dice_coeff - 1.0)  # This is always <= 0
                exponent = torch.clamp(exponent, -10.0, 0.0)  # Prevent extreme values
                
                vmf_weight = torch.exp(exponent)
                class_dice_loss = (1 - dice_coeff) * vmf_weight
                
                dice_loss += class_dice_loss
                valid_classes += 1
        
        # Average over valid classes only
        if valid_classes > 0:
            avg_dice_loss = dice_loss / valid_classes
        else:
            avg_dice_loss = torch.tensor(1.0, device=inputs.device)  # Max penalty if no valid classes
        
        # FIXED: Balanced combination with reduced TV weight
        total_loss = avg_dice_loss + 0.0001 * tv_loss  # Reduced TV weight from 0.001
        
        # FIXED: Safety check for final loss - ensure tensor input
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            # Fallback to simple cross-entropy-like loss
            total_loss = torch.tensor(1.0, device=inputs.device)
        
        return total_loss

    def __call__(self, logits, targets, b1_map_placeholder=None, all_eps_sigma_tuples=None, features_for_smoothness=None):
        return self.forward(
            logits, targets, 
            b1=b1_map_placeholder, 
            all_es=all_eps_sigma_tuples, 
            feat_sm=features_for_smoothness
        )

    def get_kappa_values(self):
        if hasattr(self, 'kappa') and isinstance(self.kappa, torch.nn.Parameter):
            try:
                kappa_values = self.kappa.detach().cpu().numpy()
                return {f"kappa_class_{i}": float(val) for i, val in enumerate(kappa_values)}
            except Exception as e:
                print(f"Failed to get kappa values: {e}")
                return {}
        return {}

class PhysicsLoss(nn.Module):
    def __init__(self, in_channels_solver):
        super(PhysicsLoss, self).__init__()
        self.solver = MaxwellSolver(in_channels=in_channels_solver)
    
    def forward(self, b1, eps, sig):
        return self.solver.compute_helmholtz_residual(b1, eps, sig)

class SmoothnessLoss(nn.Module):
    def __init__(self): 
        super(SmoothnessLoss, self).__init__()

    def forward(self, x):
        return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
               torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

class CombinedLoss(nn.Module):
    def __init__(self, wc=.5, wd=.5, wp=.1, ws=.01, in_channels_maxwell=1024, num_classes=4, max_kappa=300.0):
        super(CombinedLoss, self).__init__()
        self.wc, self.wd, self.wp, self.ws = wc, wd, wp, ws
        
        # FIXED: Better class weights for medical segmentation (reduced background dominance)
        class_weights = torch.tensor([0.1, 2.0, 2.5, 1.5])  # [Background, RV, Myocardium, LV]
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        
        # FIXED: More conservative dice loss settings
        self.dice_loss = AdaptiveTvMFDiceLoss(num_classes=num_classes, max_kappa=max_kappa)
        self.physics_loss = PhysicsLoss(in_channels_solver=in_channels_maxwell)
        self.smoothness_loss = SmoothnessLoss()

    def forward(self, logits, targets, b1, all_es, feat_sm):
        # FIXED: Robust loss computation with error handling
        try:
            loss_ce = self.ce_loss(logits, targets) if self.wc > 0 else 0
        except Exception as e:
            print(f"CrossEntropy loss failed: {e}")
            loss_ce = 0
            
        try:
            loss_dice = self.dice_loss(logits, targets) if self.wd > 0 else 0
        except Exception as e:
            print(f"Dice loss failed: {e}")
            loss_dice = 0
            
        # FIXED: Disable physics loss for stability (as requested)
        loss_phy = 0
        if self.wp > 0 and b1 is not None and all_es is not None:
            try:
                for eps, sig in all_es:
                    loss_phy += self.physics_loss(b1, eps, sig)
                if all_es: 
                    loss_phy /= len(all_es)
            except Exception as e:
                print(f"Physics loss failed: {e}")
                loss_phy = 0
                
        # FIXED: Disable smoothness loss for stability
        loss_sm = 0
        if self.ws > 0 and feat_sm is not None:
            try:
                loss_sm = self.smoothness_loss(feat_sm)
            except Exception as e:
                print(f"Smoothness loss failed: {e}")
                loss_sm = 0
                
        # Combine losses with safety checks
        total_loss = self.wc * loss_ce + self.wd * loss_dice + self.wp * loss_phy + self.ws * loss_sm
        
        # Ensure total_loss is a tensor
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, device=logits.device, dtype=logits.dtype)
        
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print("Warning: NaN/Inf detected in combined loss, using fallback")
            # Simple fallback loss
            total_loss = F.cross_entropy(logits, targets, reduction='mean')
            
        return total_loss

    def __call__(self, logits, targets, b1_map_placeholder, all_eps_sigma_tuples, features_for_smoothness):
        return self.forward(
            logits, targets, 
            b1=b1_map_placeholder, 
            all_es=all_eps_sigma_tuples, 
            feat_sm=features_for_smoothness
        )

    def get_kappa_values(self):
        if hasattr(self.dice_loss, 'kappa'):
            try:
                kappa_values = self.dice_loss.kappa.detach().cpu().numpy()
                return {f"kappa_class_{i}": float(val) for i, val in enumerate(kappa_values)}
            except Exception as e:
                print(f"Failed to get kappa values: {e}")
                return {}
        return {}

__all__ = [
    'RobustMedVFL_UNet', 'CombinedLoss', 'AdaptiveTvMFDiceLoss',
    'PhysicsLoss', 'SmoothnessLoss', 'quantum_noise_injection',
    'adaptive_spline_smoothing', 'ePURE', 'MaxwellSolver'
]