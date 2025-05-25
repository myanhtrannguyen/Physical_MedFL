import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
import h5py
from skimage.transform import resize
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import math

# --- Configuration ---
NUM_EPOCHS_CENTRALIZED = 200
NUM_CLASSES = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
BATCH_SIZE = 4

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order # Used here directly

        # The grid calculation seems plausible, keep it.
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            # The size of the last dimension should be grid_size + spline_order
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # reset_parameters calls curve2coeff, which is now a placeholder
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # This noise is used to initialize spline_weight via curve2coeff
            # Need to generate dummy 'y' for curve2coeff placeholder
            # The original curve2coeff expects y shape (batch_size, in_features, out_features)
            # In reset_parameters, the 'batch_size' is effectively grid_size + 1 points sampled on the grid
            dummy_x_for_coeff_init = self.grid.T[self.spline_order : -self.spline_order] # Shape [grid_size + 1, in_features]
            
            # Dummy y for curve2coeff needs shape [grid_size + 1, in_features, out_features]
            # Original noise shape: (grid_size + 1, in_features, out_features) - This shape seems correct for a dummy y
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    dummy_x_for_coeff_init, # x for curve2coeff
                    noise, # y for curve2coeff
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)


    # --- Placeholder methods for B-spline calculation ---
    # Needs correct implementation
    # def b_splines(self, x: torch.Tensor):
    #     """
    #     *** PLACEHOLDER ***
    #     Correct B-spline bases computation is required here.
    #     Currently returns a dummy tensor of the expected final shape.
    #     Expected input x shape: (batch_size, in_features).
    #     Expected output shape: (batch_size, in_features, grid_size + spline_order).
    #     """
    #     assert x.dim() == 2 and x.size(1) == self.in_features, f"Placeholder b_splines expected input dim 2 with size {self.in_features}, but got {x.shape}"
    #     expected_shape = (x.size(0), self.in_features, self.grid_size + self.spline_order)
    #     # Return dummy tensor (e.g., ones) of the expected shape and device/dtype
    #     return torch.ones(expected_shape, dtype=x.dtype, device=x.device)

    def b_splines(self, x: torch.Tensor):
        # Kiểm tra shape input như trong code gốc của bạn
        assert x.dim() == 2 and x.size(1) == self.in_features, \
            f"b_splines expected input dim 2 with size {self.in_features}, but got {x.shape}"
        num_bases = self.grid_size + self.spline_order
        frequencies = torch.arange(num_bases, dtype=x.dtype, device=x.device) * (2 * math.pi / num_bases)
        x_expanded = x.unsqueeze(-1)
        frequencies_reshaped = frequencies.view(1, 1, num_bases)
        sin_input = x_expanded * frequencies_reshaped
        basis_output = torch.sin(sin_input)
        expected_shape = (x.size(0), self.in_features, num_bases)
        assert basis_output.shape == expected_shape, \
            f"b_splines output shape mismatch: expected {expected_shape}, got {basis_output.shape}"
        return basis_output

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, y: torch.Tensor = None, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    # --- End Placeholder methods ---

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        # Input shape is expected to be [N, in_features] for KANLinear (N is batch*H*W in ConvKANBlock)
        assert x.dim() == 2 and x.size(1) == self.in_features, f"Expected input dim 2 with size {self.in_features}, but got {x.shape}"

        # Ensure tensors are on the same device
        x = x.to(self.base_weight.device) # Use parameter device as reference

        # The base part remains the same
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # The spline part uses b_splines and scaled_spline_weight
        # b_splines expects [N, in_features] and returns [N, in_features, grid_size + spline_order]
        spline_bases = self.b_splines(x)

        # Reshape spline_bases and scaled_spline_weight for F.linear
        # spline_bases_flat: [N, in_features * (grid_size + spline_order)]
        # scaled_spline_weight_flat: [out_features, in_features * (grid_size + spline_order)]
        spline_output = F.linear(
            spline_bases.view(x.size(0), -1), # Flatten last two dims of bases
            self.scaled_spline_weight.view(self.out_features, -1), # Flatten last two dims of weights
        )
        return base_output + spline_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.
        Uses the currently stored spline_weight (which might be dummy if never updated).
        """
        # Ensure weight is on the correct device
        # Using detach() to avoid computing gradients through regularization calculation itself
        l1_fake = self.spline_weight.abs().mean(-1) #.detach() # [out_features, in_features]

        # Sum of L1 norms across all input-output pairs
        regularization_loss_activation = l1_fake.sum()

        # Entropy of normalized L1 norms
        # Add epsilon for numerical stability if sum is zero
        sum_l1 = regularization_loss_activation.sum() + 1e-6
        p = l1_fake / sum_l1 # Normalize to get probabilities [out, in]
        # Calculate entropy for each (out, in) pair and sum them up
        # Add epsilon for numerical stability in log
        regularization_loss_entropy = -torch.sum(p * torch.log(p + 1e-6))

        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

# The provided KANConv seems to be a wrapper for sequence of KANLinear.
# In our U-Net context, we want a CONVOLUTIONAL KAN block.
# Let's create a new block that combines Conv2d and KANLinear.
class ConvKANBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        **kan_params # All KANLinear specific params should be in kan_params dict
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # Apply KANLinear to the feature channels after convolution
        # This means input features to KANLinear are the output channels of Conv2d
        # Pass **kan_params directly to KANLinear. grid_size, spline_order, etc. will be taken from here.
        self.kan_linear = KANLinear(
            out_channels,
            out_channels, # Assuming KAN is applied to map channels -> channels per spatial location
            **kan_params # Pass all KAN parameters
        )

        # Store the output of the Conv2d layer (input to KANLinear after reshape)
        # This is used by update_kan_grid
        self._features_for_grid_update = None

    def forward(self, x):
        # Ensure input is float
        x_float = x.float()
        conv_out = self.conv(x_float) # [B, C_out, H, W]

        # Store features for grid update (sample a batch or aggregate over batches)
        # Store detached features. Only need features from Conv2d output distribution.
        if self.training:
             self._features_for_grid_update = conv_out.detach().clone()

        B, C, H, W = conv_out.shape

        # Reshape for KANLinear: [B, C, H, W] -> [B*H*W, C]
        conv_out_flat = conv_out.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Apply KANLinear: [B*H*W, C] -> [B*H*W, C]
        kan_out_flat = self.kan_linear(conv_out_flat)

        # Reshape back: [B*H*W, C] -> [B, C, H, W]
        kan_out = kan_out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return kan_out # The combined Conv + KAN activation effect

    @torch.no_grad()
    def update_kan_grid(self, margin=0.01):
         # Update KAN grid using the stored features from Conv2d output
         # This calls the placeholder update_grid in KANLinear
         if self._features_for_grid_update is not None and self._features_for_grid_update.numel() > 0:
            B, C, H, W = self._features_for_grid_update.shape
            features_flat = self._features_for_grid_update.permute(0, 2, 3, 1).reshape(B * H * W, C)
            self.kan_linear.update_grid(features_flat, margin=margin)

            # Clear stored features after update
            self._features_for_grid_update = None
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
        # Ensure input is float
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

    smoothed = TF.gaussian_blur(x_float, kernel_size=kernel_size_tuple, sigma=sigma_tuple)

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

# --- Model Components (U-Net based) ---
class SplineEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spline_degree=3, kan_params=None):
        super().__init__()
        if kan_params is None: kan_params = {}

        # Use the implemented ePURE and adaptive_spline_smoothing
        # ePURE input channels should match the input of this encoder block
        self.noise_estimator = ePURE(in_channels=in_channels)

        # Replace KANConvPlaceholder with the new ConvKANBlock
        # Pass **kan_params directly.
        self.conv_kan1 = ConvKANBlock(in_channels, out_channels, kernel_size=3, **kan_params)
        self.conv_kan2 = ConvKANBlock(out_channels, out_channels, kernel_size=3, **kan_params)

    def forward(self, x):
        # Ensure input is float for ePURE and smoothing
        x_float = x.float()

        noise_profile = self.noise_estimator(x_float) # Estimate noise profile
        x_smoothed = adaptive_spline_smoothing(x_float, noise_profile) # Apply adaptive smoothing

        # Apply ConvKAN blocks
        x = self.conv_kan1(x_smoothed)
        x = self.conv_kan2(x)
        return x
    
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
    
# Locate the MaxwellSolver class
class MaxwellSolver(nn.Module):
    def __init__(self, in_channels, hidden_dim=32):
        super(MaxwellSolver, self).__init__()
        # Keep original implementation
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)) # Output 2 channels: eps, sigma
        # Constants for k0 - assuming relevant frequency (e.g., Larmor freq for 1T field ~42.58 MHz)
        omega = 2 * np.pi * 42.58e6 # Angular frequency
        mu_0 = 4 * np.pi * 1e-7 # Permeability of free space (H/m)
        eps_0 = 8.854187817e-12 # Permittivity of free space (F/m)
        # Wave number k0 = omega * sqrt(mu0 * eps0)
        self.k0 = torch.tensor(omega * np.sqrt(mu_0 * eps_0), dtype=torch.float32)

    def forward(self, x):
        # Input x is concatenated features (skip + upsampled) [B, concat_ch, H, W]
        # Output eps_sigma_map is [B, 2, H, W] where channels are epsilon and sigma
        # Ensure input is float
        x_float = x.float()
        eps_sigma_map = self.encoder(x_float)
        # Ensure output is float
        return eps_sigma_map[:, 0:1, :, :].float(), eps_sigma_map[:, 1:2, :, :].float() # Return (eps, sigma)


    def compute_helmholtz_residual(self, b1_map, eps, sigma):
        # Keep original implementation but ensure tensor types and devices match
        # Move k0 to the device of the input tensors
        device = b1_map.device
        self.k0 = self.k0.to(device)
        omega = 2 * np.pi * 42.58e6 # Ensure omega is defined if needed here again, or use the same constant

        # Ensure b1_map, eps, sigma are float and on the same device
        b1_map_float = b1_map.float().to(device)
        eps_float = eps.float().to(device)
        sigma_float = sigma.float().to(device)

        # If b1_map is not complex, make it complex with zero imaginary part
        b1_map_complex = torch.complex(b1_map_float, torch.zeros_like(b1_map_float)) if not b1_map_float.is_complex() else b1_map_float

        # Ensure spatial dimensions match for calculation
        # If b1_map was resized already, its size should match eps/sigma
        size = eps_float.shape[2:] # Use eps/sigma size as the target size

        # Interpolate B1 map only if sizes don't match - this should ideally not happen
        # if b1_map_list is prepared correctly, but adds robustness.
        # Ensure B1 map has a channel dimension [B, C, H, W] before checking/interpolating size
        if b1_map_complex.dim() == 3: # Assume [B, H, W], add channel dim
             b1_map_complex = b1_map_complex.unsqueeze(1) # -> [B, 1, H, W]
        elif b1_map_complex.dim() != 4:
             print(f"Warning: Unexpected B1 map dimension {b1_map_complex.dim()}. Skipping Helmholtz residual.")
             return torch.zeros_like(eps_float) # Return zero residual if B1 map shape is wrong


        if b1_map_complex.shape[2:] != size:
             # print(f"Warning: B1 map size {b1_map_complex.shape[2:]} doesn't match physics output size {size}. Resizing B1 map.")
             # Interpolate real and imag parts separately and combine for complex
             b1_map_real_interp = F.interpolate(b1_map_complex.real, size=size, mode='bilinear', align_corners=False)
             b1_map_imag_interp = F.interpolate(b1_map_complex.imag, size=size, mode='bilinear', align_corners=False)
             b1_map_complex = torch.complex(b1_map_real_interp, b1_map_imag_interp)


        # Compute Laplacian of B1 map
        # CORRECTED: Removed .unsqueeze(1) and .squeeze(1) calls from _laplacian_2d input
        # assuming x_complex.real/imag already have [B, C, H, W] shape where C is 1.
        # (The unsqueeze(1) added an extra dim, leading to the 5D error).
        lap_b1 = self._laplacian_2d(b1_map_complex)

        # Compute Helmholtz equation residual: nabla^2 B1 + k0^2 * eps_c * B1
        # Ensure k0^2 has correct complex type if needed, but multiplying by complex tensor handles it
        res = lap_b1 + (self.k0 ** 2) * torch.complex(eps_float, -sigma_float / omega) * b1_map_complex # Recomputed eps_c inline

        # Return sum of squared real and imaginary parts of the residual
        return res.real ** 2 + res.imag ** 2

    def _laplacian_2d(self, x_complex):
        # Keep original implementation, ensure kernel is on device and dtype matches
        # x_complex shape is assumed to be [B, C, H, W] where C is 1 for real/imag parts
        device = x_complex.device
        dtype = x_complex.real.dtype
        k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=device, dtype=dtype).reshape(1,1,3,3)

        # Apply convolution separately to real and imaginary parts
        # The input to F.conv2d should be [B, C_in, H_in, W_in].
        # x_complex.real/imag are [B, 1, H, W] based on how b1_map is handled.
        # CORRECTED: Removed .unsqueeze(1) and .squeeze(1)
        rl = F.conv2d(x_complex.real, k, padding=1)
        im = F.conv2d(x_complex.imag, k, padding=1)

        return torch.complex(rl, im)
    
class PhysicsKANDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kan_params=None):
        super().__init__()
        if kan_params is None: kan_params = {}

        # Upsampling layer
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Number of channels after concatenation
        concat_ch = in_channels // 2 + skip_channels

        # Maxwell Solver takes concatenated features
        self.maxwell_solver = MaxwellSolver(concat_ch)

        # Replace KANConvPlaceholder with ConvKANBlock
        # Pass **kan_params directly.
        self.conv_kan1 = ConvKANBlock(concat_ch, out_channels, kernel_size=3, **kan_params)
        self.conv_kan2 = ConvKANBlock(out_channels, out_channels, kernel_size=3, **kan_params)

    def forward(self, x, skip_connection):
        # Ensure inputs are float
        x_float = x.float()
        skip_connection_float = skip_connection.float()

        x_up = self.up(x_float)

        # Pad upsampled tensor to match skip connection size if needed
        # Ensure padding values are non-negative
        diffY = max(0, skip_connection_float.size()[2] - x_up.size()[2])
        diffX = max(0, skip_connection_float.size()[3] - x_up.size()[3])

        # Pad using calculated differences
        # Ensure padding is applied correctly for potentially smaller upsampled feature maps
        # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection and upsampled tensor
        # Ensure spatial sizes match exactly after padding before concatenating
        assert x_up.shape[2:] == skip_connection_float.shape[2:], \
            f"Spatial size mismatch after padding: upsampled {x_up.shape[2:]} vs skip {skip_connection_float.shape[2:]}"
        x_cat = torch.cat([skip_connection_float, x_up], dim=1)

        # Predict physics maps using Maxwell Solver
        # MaxwellSolver takes x_cat features, outputs eps/sigma at x_cat's size
        es_tuple = self.maxwell_solver(x_cat) # Returns (epsilon, sigma)

        # Apply ConvKAN blocks to the concatenated features
        out = self.conv_kan2(self.conv_kan1(x_cat))

        # Return output features and physics maps
        return out, es_tuple
    
class RobustMedVFL_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, kan_params=None):
        super().__init__()
        if kan_params is None: kan_params = {}

        # Pass kan_params to all blocks using ConvKANBlock
        self.enc1 = SplineEncoderBlock(n_channels, 64, kan_params=kan_params)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = SplineEncoderBlock(64, 128, kan_params=kan_params)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = SplineEncoderBlock(128, 256, kan_params=kan_params)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = SplineEncoderBlock(256, 512, kan_params=kan_params)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck using SplineEncoderBlock
        self.bottleneck = SplineEncoderBlock(512, 1024, kan_params=kan_params)

        # Decoder path using PhysicsKANDecoderBlock
        self.dec1 = PhysicsKANDecoderBlock(1024, 512, 512, kan_params=kan_params)
        self.dec2 = PhysicsKANDecoderBlock(512, 256, 256, kan_params=kan_params)
        self.dec3 = PhysicsKANDecoderBlock(256, 128, 128, kan_params=kan_params)
        self.dec4 = PhysicsKANDecoderBlock(128, 64, 64, kan_params=kan_params)

        # Output convolution for segmentation
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Ensure input is float
        x = x.float()

        # Encoder
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with skip connections and physics outputs
        d1, es1 = self.dec1(b, e4) # es1 = (eps, sigma) at dec1 concat size
        d2, es2 = self.dec2(d1, e3) # es2 = (eps, sigma) at dec2 concat size
        d3, es3 = self.dec3(d2, e2) # es3 = (eps, sigma) at dec3 concat size
        d4, es4 = self.dec4(d3, e1) # es4 = (eps, sigma) at dec4 concat size

        # Final segmentation output
        logits = self.out_conv(d4)

        # Feature map to use for smoothness loss
        feat_sm = d4

        # Return logits and all physics outputs from decoder blocks as a tuple of tuples
        return logits, (es1, es2, es3, es4), feat_sm

    @torch.no_grad()
    def update_kan_grids(self, dataloader, num_batches=5, margin=0.01):
        """
        Update KAN grids in all ConvKANBlocks using a few batches from the dataloader.
        Calls placeholder update_grid in KANLinear.
        """
        self.eval() # Set model to eval mode temporarily
        # print(f"Updating KAN grids using {num_batches} batches...") # Keep print for visibility

        # Check if dataloader is valid and has data
        if dataloader is None or len(dataloader.dataset) == 0:
            # print("No data available for KAN grid update.") # Keep print for visibility
            self.train() # Set model back to train mode
            return

        # Iterate through a few batches to trigger storing features in ConvKANBlocks
        batches_processed = 0
        # Use a temporary loader for sampling, ensure no shuffle for reproducible sampling if needed
        # Use a try-except block in case the dataloader cannot be iterated
        try:
            temp_loader = DataLoader(dataloader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

            for imgs, _ in temp_loader:
                if batches_processed >= num_batches:
                    break

                # Move data to model's device
                imgs = imgs.to(next(self.parameters()).device)

                # Run forward pass to trigger storing features in ConvKANBlocks' _features_for_grid_update
                with torch.no_grad():
                    # Call forward pass. We don't need the outputs here, just the side effect of storing features.
                    _ = self.forward(imgs) # Call the model's forward method

                batches_processed += 1

        except Exception as e:
            print(f"Error during KAN grid update data pass: {e}")
            # Continue to attempt update even if data pass failed partially


        # Now iterate through all ConvKANBlock instances and call their update_kan_grid method
        # This calls the placeholder update_grid in KANLinear
        kan_blocks_updated = 0
        for module in self.modules():
            if isinstance(module, ConvKANBlock):
                module.update_kan_grid(margin=margin)
                kan_blocks_updated += 1

        # print(f"Attempted grid update for {kan_blocks_updated} ConvKANBlocks.") # Keep print for visibility
        # print("KAN grid update finished (using placeholder logic).") # Keep print for visibility
        self.train() # Set model back to train model

# --- Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-6):
        super().__init__(); self.num_classes, self.smooth = num_classes, smooth
    def forward(self, inputs, targets):
        # Ensure inputs and targets are on the same device
        inputs = inputs.to(targets.device) # Move inputs to targets device

        inputs = F.softmax(inputs, dim=1); loss = 0
        for i in range(self.num_classes):
            # Ensure tensors are float for calculation and on the same device
            inp_c = inputs[:,i,:,:].contiguous().view(-1).float()
            tgt_c = (targets==i).float().contiguous().view(-1).float()
            inter = (inp_c * tgt_c).sum()
            union = inp_c.sum() + tgt_c.sum()
            dice = (2.*inter+self.smooth)/(union+self.smooth)
            loss += (1-dice)
        return loss / self.num_classes

class PhysicsLoss(nn.Module):
    def __init__(self): # Removed in_channels_solver as it's not used here
        super().__init__();
        # MaxwellSolver instance is inside the decoder blocks, accessed via the model

    def forward(self, b1_map_list, all_es_list, model):
        # Ensure inputs are valid
        if not all_es_list or len(all_es_list) == 0:
             # print("Warning: No physics outputs provided.")
             # Ensure return tensor is on the device of model parameters if possible
             device = next(model.parameters(), torch.empty(0)).device # Get device or default
             return torch.tensor(0., device=device) # Return 0 loss if no physics outputs

        # Get PhysicsKANDecoderBlock instances in the order they appear in the model
        decoder_blocks = [module for module in model.modules() if isinstance(module, PhysicsKANDecoderBlock)]

        # Ensure the number of blocks matches the number of physics outputs and b1 maps
        if len(decoder_blocks) != len(all_es_list) or len(decoder_blocks) != (len(b1_map_list) if b1_map_list is not None else -1):
             # print(f"Warning: Mismatch: {len(decoder_blocks)} decoder blocks, {len(all_es_list)} physics outputs, {len(b1_map_list) if b1_map_list is not None else 'None'} b1 maps. Physics loss might be incorrect.")
             # Return 0 loss if mismatch
             device = next(model.parameters(), torch.empty(0)).device
             return torch.tensor(0., device=device)

        physics_loss = torch.tensor(0., device=all_es_list[0][0].device) # Use device of physics output

        # Iterate through decoder blocks and their corresponding physics outputs and b1 maps
        for i in range(len(decoder_blocks)):
            decoder_block = decoder_blocks[i]
            # Ensure b1 map is valid and on device. Assume b1_map_list is list of tensors.
            b1 = b1_map_list[i]
            if b1 is None:
                 # print(f"Warning: Skipping physics loss for decoder block {i} due to missing B1 map.")
                 continue # Skip this block if B1 map is None

            b1 = b1.to(b1.device) # Ensure b1 map is on its original device (should match physics output)
            eps, sig = all_es_list[i] # eps and sig are already on device from model output

            # Ensure eps and sig are valid tensors
            if eps is None or sig is None:
                 # print(f"Warning: Skipping physics loss for decoder block {i} due to missing eps/sig.")
                 continue # Skip if physics outputs are missing

            # Compute residual using the MaxwellSolver instance within the decoder block
            # compute_helmholtz_residual expects b1, eps, sig on the same device
            try:
                residual = decoder_block.maxwell_solver.compute_helmholtz_residual(b1, eps, sig)
                # Add the mean of the residual squared to the total physics loss
                physics_loss += torch.mean(residual)
            except Exception as e:
                 print(f"Error computing Helmholtz residual for block {i}: {e}")
                 # Continue loop even if one residual computation fails

        # Average physics loss over the *successful* computations
        num_successful = len([b for b in b1_map_list if b is not None]) # Count how many B1 maps were valid
        # For simplicity, average over the number of decoder blocks if at least one was successful
        return physics_loss / len(decoder_blocks) if len(decoder_blocks) > 0 and physics_loss > 0 else torch.tensor(0., device=physics_loss.device)

class SmoothnessLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        # Ensure input is float and on device
        x = x.float()
        device = x.device

        # Compute absolute differences in spatial dimensions
        # Handle cases where spatial dims are too small for differencing
        if x.size(2) < 2 or x.size(3) < 2:
            # print("Warning: Feature map too small for smoothness loss.")
            return torch.tensor(0., device=device)

        # Ensure differences are on the same device
        dy = torch.abs(x[:,:,1:,:]-x[:,:,:-1,:])
        dx = torch.abs(x[:,:,:,1:]-x[:,:,:,:-1])
        # Mean of differences
        return torch.mean(dy) + torch.mean(dx)
    
class CombinedLoss(nn.Module):
    def __init__(self, wc=.5, wd=.5, wp=.1, ws=.01, w_kan_reg=0.001, num_classes=4, in_channels_maxwell=1024): # Removed unused icm
        super().__init__()
        self.ce = nn.CrossEntropyLoss() # Cross Entropy Loss
        self.dl = DiceLoss(num_classes) # Dice Loss
        self.pl = PhysicsLoss() # Physics Loss (doesn't need icm)
        self.sl = SmoothnessLoss() # Smoothness Loss

        self.wc, self.wd, self.wp, self.ws, self.w_kan_reg = wc, wd, wp, ws, w_kan_reg
        self.num_classes = num_classes # Keep if used elsewhere

    # Added 'model' and 'b1_map_list' arguments
    # b1_map_list should be a list/tuple of B1 maps (or intensity proxies)
    # corresponding to the levels where physics outputs (all_es) are generated.
    def forward(self, logits, targets, b1_map_list, all_es, model, feat_sm):
        # Ensure logits and targets are on the same device before calculation
        device = logits.device
        targets = targets.to(device)

        # Ensure targets are long for CrossEntropyLoss
        targets_long = targets.long()

        # Segmentation Losses
        lce = self.ce(logits, targets_long)
        ldc = self.dl(logits, targets_long) # DiceLoss handles device internally

        # Initial total loss from segmentation
        loss = self.wc * lce + self.wd * ldc

        # Physics Loss
        # Pass b1_map_list, all_es tuple, and the model instance to PhysicsLoss
        # PhysicsLoss internally checks for valid inputs and lengths
        lphy = self.pl(b1_map_list, all_es, model)
        loss += self.wp * lphy

        # Smoothness Loss (currently not active as feat_sm=None in client)
        lsm = torch.tensor(0., device=device)
        if feat_sm is not None:
            lsm = self.sl(feat_sm) # SmoothnessLoss handles device internally
            loss += self.ws * lsm

        # KAN Regularization Loss
        lkan_reg = torch.tensor(0., device=device)
        # Calculate KAN regularization loss by iterating through model modules
        # This calls the regularization_loss method in KANLinear (which is not a placeholder)
        if model is not None and self.w_kan_reg > 0:
            for module in model.modules():
                if isinstance(module, ConvKANBlock):
                    # Sum regularization loss from all KANLinear layers within ConvKANBlocks
                    # Ensure the regularization loss is on the correct device before summing
                    lkan_reg += module.kan_linear.regularization_loss().to(device)
            loss += self.w_kan_reg * lkan_reg

        # You might want to return individual loss components for monitoring
        # return loss, lce, ldc, lphy, lsm, lkan_reg
        return loss
    
# --- Data Loading ---
def load_h5_data(directory, is_training=True, target_size=(256,256), max_samples=None):
    imgs, msks, count = [], [], 0
    if not os.path.exists(directory):
        return np.array([]), (np.array([]) if is_training else None)

    for fname in sorted(os.listdir(directory)):
        if max_samples and count >= max_samples:
            break
        if fname.endswith('.h5'):
            try:
                with h5py.File(os.path.join(directory, fname), 'r') as f:
                    img_d = f['image'][:]
                    msk_d = f['label'][:] if 'label' in f else None  # <- luôn cố load label nếu có

                    proc = lambda d, t_sz, is_m: resize(
                        d.astype(np.uint8 if is_m else np.float32), t_sz,
                        order=(0 if is_m else 1), preserve_range=True,
                        anti_aliasing=(not is_m), mode='reflect'
                    ).astype(np.uint8 if is_m else np.float32)

                    if img_d.ndim == 3:
                        for i in range(img_d.shape[0]):
                            imgs.append(np.expand_dims(proc(img_d[i], target_size, False), axis=-1))
                            if msk_d is not None:
                                msks.append(proc(msk_d[i], target_size, True))
                    elif img_d.ndim == 2:
                        imgs.append(np.expand_dims(proc(img_d, target_size, False), axis=-1))
                        if msk_d is not None:
                            msks.append(proc(msk_d, target_size, True))
                count += 1
            except Exception as e:
                print(f"Err load {fname}: {e}")

    im_np = np.array(imgs, dtype=np.float32) if imgs else np.empty((0, target_size[0], target_size[1], 1), dtype=np.float32)
    msk_np = np.array(msks, dtype=np.uint8) if msks else None  # <- đơn giản hóa
    return im_np, msk_np

# --- Metrics ---
def evaluate_metrics(model, dataloader, device, num_classes=4):
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
            logits,_,_ = model(imgs)
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
    
    # --- Load Data ---
    base_data_path = '/kaggle/input/acdc-dataset/ACDC_preprocessed'
    if not os.path.exists(base_data_path) or not os.listdir(base_data_path):
        print(f"Path '{base_data_path}' not found/empty. Using DUMMY data.")
        # Tạo dữ liệu dummy để code chạy được
        X_train_tensor = torch.randn(100, 1, IMG_SIZE, IMG_SIZE) # 100 mẫu huấn luyện
        y_train_tensor = torch.randint(0, NUM_CLASSES, (100, IMG_SIZE, IMG_SIZE))
        X_val_tensor = torch.randn(20, 1, IMG_SIZE, IMG_SIZE) # 20 mẫu validation
        y_val_tensor = torch.randint(0, NUM_CLASSES, (20, IMG_SIZE, IMG_SIZE))
    else:
        train_dir = os.path.join(base_data_path, 'ACDC_training_slices')
        test_dir = os.path.join(base_data_path, 'ACDC_testing_volumes') # Nếu cần test set riêng
        
        # Tải toàn bộ dữ liệu huấn luyện
        all_train_images_np, all_train_masks_np = load_h5_data(train_dir, is_training=True, target_size=(IMG_SIZE, IMG_SIZE), max_samples=600) # Giảm max_samples cho nhanh
        all_test_images_np, all_test_masks_np = load_h5_data(test_dir, is_training=False, target_size=(IMG_SIZE, IMG_SIZE), max_samples=200)
        
        if all_train_images_np.size == 0:
            raise ValueError("Training data is empty after loading. Check data path and content.")
        
        # Normalize
        if np.max(all_train_images_np) > 0:
            all_train_images_np = all_train_images_np / np.max(all_train_images_np)

        if np.max(all_test_images_np) > 0:
            all_test_images_np = all_test_images_np / np.max(all_test_images_np)
        
        X_test_tensor = torch.tensor(all_test_images_np).permute(0, 3, 1, 2).float()
        y_test_tensor = torch.tensor(all_test_masks_np).long()
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
        
        # Chia train/validation từ toàn bộ dữ liệu đã tải
        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            all_train_images_np, all_train_masks_np, test_size=0.2, random_state=42 # 20% cho validation
        )
        
        X_train_tensor = torch.tensor(X_train_np).permute(0, 3, 1, 2).float()
        y_train_tensor = torch.tensor(y_train_np).long()
        X_val_tensor = torch.tensor(X_val_np).permute(0, 3, 1, 2).float()
        y_val_tensor = torch.tensor(y_val_np).long()

    if len(X_train_tensor) == 0: raise ValueError("No training samples after split.")
    if len(X_val_tensor) == 0: print("Warning: Validation set is empty after split.")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print("Data loaded and prepared for centralized training.")

    # --- Initialize Model, Criterion, Optimizer ---
    model = RobustMedVFL_UNet(n_channels=1, n_classes=NUM_CLASSES).to(DEVICE)
    # model = RobustMedVFL_UNet(n_channels=1, n_classes=NUM_CLASSES)

    # if DEVICE.type == 'cuda' and torch.cuda.device_count() > 1:
    #     print(f"Có {torch.cuda.device_count()} GPU! Sử dụng Data Parallel.")
    #     model = nn.DataParallel(model)

    # model.to(DEVICE)
    
    criterion = CombinedLoss(num_classes=NUM_CLASSES, in_channels_maxwell=1024).to(DEVICE)
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
            logits, all_eps_sigma_tuples, feat_sm = model(images_noisy)
            b1_map_placeholder = torch.randn_like(images[:, 0:1, ...], device=DEVICE) # Placeholder
            
            # loss = criterion(logits, targets, b1_map_placeholder, all_eps_sigma_tuples) #, features_for_smoothness=None)
            loss = criterion(logits, targets, b1_map_placeholder, all_eps_sigma_tuples, model=model, feat_sm=feat_sm)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
        
            avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
        print(f"  Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_dataloader.dataset and len(val_dataloader.dataset) > 0:
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
            print("  Validation dataset is empty. Skipping validation.")

    print("\n--- Centralized Training Finished ---")

# --- Evaluate on Test Set ---
if 'test_dataloader' in locals() and len(test_dataloader.dataset) > 0:
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