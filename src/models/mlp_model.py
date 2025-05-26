"""
Hello m, t đã cop y hệt file Kaggle xuống, h nó đang conflict khá là nhiều
t đang fix lại,
Bây h t cần m xem là tỏng model có phần nào m cần chuẩn hoá không, vì code AI
cũng ngu vl
Xem lại giúp bạn nhé
"""

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
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_bn)]
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
    def __init__(self, num_classes=4, smooth=1e-6): # Đã dùng num_classes
        super().__init__(); self.num_classes, self.smooth = num_classes, smooth
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1); loss = 0
        for i in range(self.num_classes):
            inp_c = inputs[:,i,:,:].contiguous().view(-1)
            tgt_c = (targets==i).float().contiguous().view(-1)
            inter = (inp_c * tgt_c).sum()
            dice = (2.*inter+self.smooth)/(inp_c.sum()+tgt_c.sum()+self.smooth)
            loss += (1-dice)
        return loss / self.num_classes

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
    # SỬA Ở ĐÂY: đổi 'nc' thành 'num_classes'
    def __init__(self, wc=.5, wd=.5, wp=.1, ws=.01, in_channels_maxwell=1024, num_classes=4):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dl = DiceLoss(num_classes=num_classes) # Khởi tạo DiceLoss với num_classes
        self.pl = PhysicsLoss(in_channels_solver=in_channels_maxwell) # Đổi tên cho rõ nghĩa
        self.sl = SmoothnessLoss()
        self.wc,self.wd,self.wp,self.ws = wc,wd,wp,ws
        # self.num_classes = num_classes # Có thể lưu lại nếu cần dùng ở đâu khác trong CombinedLoss

    def forward(self, logits, targets, b1, all_es, feat_sm=None):
        lce = self.ce(logits,targets.long())
        ldc = self.dl(logits,targets.long()) # DiceLoss đã có num_classes từ __init__
        loss = self.wc*lce + self.wd*ldc
        
        lphy = torch.tensor(0.,device=logits.device)
        if b1 is not None and all_es and len(all_es)>0:
            # Giả sử all_es[0] là tuple (eps, sigma) từ tầng decoder quan trọng nhất
            e1,s1 = all_es[0] 
            lphy=self.pl(b1,e1,s1)
            loss+=self.wp*lphy
            
        lsm = torch.tensor(0.,device=logits.device)
        if feat_sm is not None:
            lsm=self.sl(feat_sm)
            loss+=self.ws*lsm
        return loss
    
# # --- Data Loading ---
# def load_h5_data(directory, is_training=True, target_size=(256,256), max_samples=None):
#     imgs,msks,count = [],[],0
#     if not os.path.exists(directory): return np.array([]), (np.array([]) if is_training else None)
#     for fname in sorted(os.listdir(directory)):
#         if max_samples and count>=max_samples: break
#         if fname.endswith('.h5'):
#             try:
#                 with h5py.File(os.path.join(directory,fname),'r') as f:
#                     img_d, msk_d = f['image'][:], (f['label'][:] if is_training else None)
#                     proc = lambda d,t_sz,is_m: resize(d.astype(np.uint8 if is_m else np.float32),t_sz,order=(0 if is_m else 1),preserve_range=True,anti_aliasing=(not is_m),mode='reflect').astype(np.uint8 if is_m else np.float32)
#                     if img_d.ndim==3:
#                         for i in range(img_d.shape[0]):
#                             imgs.append(np.expand_dims(proc(img_d[i],target_size,False),axis=-1))
#                             if is_training: msks.append(proc(msk_d[i],target_size,True))
#                     elif img_d.ndim==2:
#                         imgs.append(np.expand_dims(proc(img_d,target_size,False),axis=-1))
#                         if is_training: msks.append(proc(msk_d,target_size,True))
#                 count+=1
#             except Exception as e: print(f"Err load {fname}: {e}")
#     im_np = np.array(imgs,dtype=np.float32) if imgs else np.empty((0,target_size[0],target_size[1],1),dtype=np.float32)
#     msk_np = np.array(msks,dtype=np.uint8) if is_training and msks else (np.empty((0,target_size[0],target_size[1]),dtype=np.uint8) if is_training else None)
#     return im_np, msk_np

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