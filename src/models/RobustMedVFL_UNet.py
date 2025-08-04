# --- Model Components (U-Net based) ---
from torch import nn
import torch
import torch.nn.functional as F
from src.models.ePURE import ePURE
from src.models.adaptive_spline_funct import adaptive_spline_smoothing
import numpy as np
import pytorch_lightning as L

# Lightning imports
try:
    import pytorch_lightning as L
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torchmetrics
    from typing import Any, Dict, Optional, Tuple, Union
    from pytorch_lightning.utilities.types import OptimizerLRScheduler
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("PyTorch Lightning not available. Using standard PyTorch training.")


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


# Lightning Module Integration
if LIGHTNING_AVAILABLE:
    import pytorch_lightning as pl
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torchmetrics
    
    class LightningRobustMedUNet(pl.LightningModule):
        """PyTorch Lightning wrapper for RobustMedVFL_UNet with multi-GPU support."""
        
        def __init__(
            self,
            n_channels: int = 1,
            n_classes: int = 4,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            scheduler_T_max: int = 50,
            loss_config: Optional[Dict] = None
        ):
            super().__init__()
            self.save_hyperparameters()
            
            # Initialize model
            self.model = RobustMedVFL_UNet(n_channels=n_channels, n_classes=n_classes)
            
            # Initialize loss function
            from ..utils.losses import CombinedLoss
            loss_config = loss_config or {}
            self.criterion = CombinedLoss(
                in_channels_maxwell=loss_config.get('in_channels_maxwell', 1024),
                num_classes=n_classes,
                lambda_val=loss_config.get('lambda_val', 15.0),
                initial_loss_weights=loss_config.get('initial_loss_weights', None)
            )
            
            # Store parameters for custom metrics
            self.n_classes = n_classes
            
        def _calculate_dice_score(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
            """Calculate average Dice score across all classes."""
            dice_scores = []
            for c in range(self.n_classes):
                pred_c = (preds == c).float()
                target_c = (targets == c).float()
                intersection = (pred_c * target_c).sum()
                dice = (2.0 * intersection + 1e-6) / (pred_c.sum() + target_c.sum() + 1e-6)
                dice_scores.append(dice.item())
            return sum(dice_scores) / len(dice_scores)
        
        def _calculate_iou_score(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
            """Calculate average IoU score across all classes."""
            iou_scores = []
            for c in range(self.n_classes):
                pred_c = (preds == c).float()
                target_c = (targets == c).float()
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                iou_scores.append(iou.item())
            return sum(iou_scores) / len(iou_scores)
        
        def _calculate_accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
            """Calculate pixel-wise accuracy."""
            correct = (preds == targets).float().sum()
            total = targets.numel()
            return (correct / total).item()
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
            """Forward pass through the model."""
            return self.model(x)
        
        def training_step(self, batch, batch_idx: int) -> torch.Tensor:
            """Training step with combined loss."""
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch['image']
                targets = batch['mask']
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            # Forward pass
            logits, maxwell_outputs = self(images)
            
            # Extract physics components for loss
            b1 = logits  # Use logits as B1 field approximation
            all_es = maxwell_outputs  # Maxwell solver outputs from each decoder block
            feat_sm = logits  # Use logits for smoothness regularization
            
            # Compute combined loss
            loss = self.criterion(logits, targets, b1=b1, all_es=all_es, feat_sm=feat_sm)
            
            # Calculate custom metrics for training
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            # Calculate Dice score manually
            dice_score = self._calculate_dice_score(preds, targets)
            
            # Log metrics
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train_dice', dice_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
            # Log loss component weights for monitoring
            loss_weights = self.criterion.get_current_loss_weights()
            for name, weight in loss_weights.items():
                self.log(f'train_{name}', weight, on_epoch=True, sync_dist=True)
                
            return loss
        
        def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
            """Validation step."""
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch['image']
                targets = batch['mask']
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            # Forward pass
            logits, maxwell_outputs = self(images)
            
            # Extract physics components for loss
            b1 = logits
            all_es = maxwell_outputs
            feat_sm = logits
            
            # Compute loss
            loss = self.criterion(logits, targets, b1=b1, all_es=all_es, feat_sm=feat_sm)
            
            # Convert logits to predictions
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            # Calculate custom metrics
            dice = self._calculate_dice_score(preds, targets.long())
            iou = self._calculate_iou_score(preds, targets.long())
            accuracy = self._calculate_accuracy(preds, targets.long())
            
            # Log metrics
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_dice', dice, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_iou', iou, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
            
            return loss
        
        def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            """Prediction step."""
            images = batch['image']
            logits, _ = self(images)
            return F.softmax(logits, dim=1)
        
        def configure_optimizers(self):
            """Configure optimizer and learning rate scheduler."""
            optimizer = Adam(
                self.parameters(),
                lr=getattr(self.hparams, 'learning_rate', 1e-4),
                weight_decay=getattr(self.hparams, 'weight_decay', 1e-5)
            )
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=getattr(self.hparams, 'scheduler_T_max', 50),
                eta_min=1e-7
            )
            
            # Return tuple format that Lightning expects
            return [optimizer], [scheduler]
        
        def on_train_epoch_end(self) -> None:
            """Called at the end of each training epoch."""
            # Log adaptive loss information
            kappa_info = self.criterion.get_kappa_values()
            if kappa_info:
                for key, value in kappa_info.items():
                    if isinstance(value, list):
                        for i, v in enumerate(value):
                            self.log(f'kappa_{i}', v, on_epoch=True, sync_dist=True)
                    else:
                        self.log(f'loss_{key}', value, on_epoch=True, sync_dist=True)
        
        def get_model_weights(self) -> Dict[str, torch.Tensor]:
            """Get model weights for federated learning."""
            return self.model.state_dict()
        
        def set_model_weights(self, weights: Dict[str, torch.Tensor]) -> None:
            """Set model weights for federated learning."""
            self.model.load_state_dict(weights)
        
        def get_num_examples(self) -> int:
            """Get number of training examples (for federated learning)."""
            try:
                # Use getattr to avoid type checking issues
                datamodule = getattr(self.trainer, 'datamodule', None)
                if datamodule and hasattr(datamodule, 'num_train_examples'):
                    return datamodule.num_train_examples
                
                train_dataloader = getattr(self.trainer, 'train_dataloader', None)
                if train_dataloader and hasattr(train_dataloader, 'dataset'):
                    dataset = getattr(train_dataloader, 'dataset', None)
                    if dataset:
                        return len(dataset)
                
                # Fallback to a default value
                return 1000
            except Exception:
                return 1000