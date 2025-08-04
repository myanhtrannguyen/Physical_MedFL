import torch
import torch.nn as nn
import torch.nn.functional as F
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