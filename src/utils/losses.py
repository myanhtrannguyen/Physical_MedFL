import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Dict, List, Tuple


class Adaptive_tvmf_dice_loss(nn.Module):
    """
    Advanced Adaptive t-vMF Dice Loss for medical image segmentation.

    Combines:
    - von Mises-Fisher distribution similarity
    - Adaptive kappa values per class
    - Dice coefficient optimization
    - Class-specific concentration parameters
    """

    def __init__(
            self,
            num_classes: int = 4,
            lambda_val: float = 15.0,
            kappa_values=None,
            epsilon: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_val = lambda_val
        self.epsilon = epsilon

        # Initialize kappa values (concentration parameters)
        if kappa_values is not None:
            self.register_buffer('kappa_values', torch.tensor(kappa_values, dtype=torch.float32))
        else:
            self.register_buffer('kappa_values', torch.ones(num_classes) * lambda_val)

    def update_kappa_values(self, new_kappa_values) -> None:
        """Update kappa values from server (for adaptive learning)."""
        if isinstance(new_kappa_values, (list, np.ndarray)):
            new_kappa_values = torch.tensor(new_kappa_values, dtype=torch.float32)
        device = next(self.parameters()).device
        self.kappa_values.data = new_kappa_values.to(device)

    def t_vmf_similarity(self, cos_theta, kappa):
        """
        Compute t-vMF similarity.

        Args:
            cos_theta: Cosine similarity between prediction and target
            kappa: Concentration parameter
        """
        # Ensure kappa is non-negative
        kappa = F.relu(kappa) + self.epsilon
        # t-vMF similarity: exp(kappa * (cos_theta - 1))
        return torch.exp(kappa * (cos_theta - 1))

    def compute_dice_coefficient(self, pred, target):
        """Compute Dice coefficient for binary masks."""
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        return dice

    def forward(self, inputs, targets):
        """
        Forward pass of the adaptive t-vMF Dice loss.

        Args:
            inputs: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, H, W]
        """
        # Apply softmax to get probabilities
        if inputs.dim() == 4:  # [B, C, H, W]
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = F.softmax(inputs, dim=-1)

        # Convert targets to one-hot encoding
        if targets.dim() == 3:  # [B, H, W]
            targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        else:
            targets_one_hot = targets.float()

        total_loss = 0.0
        class_losses = []

        for class_idx in range(self.num_classes):
            # Get class-specific predictions and targets
            pred_class = inputs[:, class_idx, :, :]  # [B, H, W]
            target_class = targets_one_hot[:, class_idx, :, :]  # [B, H, W]

            # Flatten for cosine similarity computation
            pred_flat = pred_class.contiguous().view(-1)
            target_flat = target_class.contiguous().view(-1)

            # Skip if no positive pixels for this class
            if torch.sum(target_flat) < self.epsilon:
                class_losses.append(torch.tensor(0.0, device=inputs.device))
                continue

            # Compute cosine similarity
            cos_theta = F.cosine_similarity(
                pred_flat.unsqueeze(0),
                target_flat.unsqueeze(0),
                dim=1,
                eps=self.epsilon
            ).squeeze()

            # Get kappa for this class
            kappa_tensor = getattr(self, 'kappa_values')
            kappa = kappa_tensor[class_idx]

            # Compute t-vMF similarity
            similarity = self.t_vmf_similarity(cos_theta, kappa)

            # Compute Dice coefficient
            dice_coeff = self.compute_dice_coefficient(pred_class, target_class)

            # Combined loss: (1 - t-vMF similarity) * (1 - Dice)
            tvmf_loss = 1.0 - similarity
            dice_loss = 1.0 - dice_coeff

            # Weighted combination
            class_loss = tvmf_loss + dice_loss
            class_losses.append(class_loss)
            total_loss += class_loss

        # Return average loss across classes
        avg_loss = total_loss / self.num_classes

        # Store individual class losses for analysis
        self.last_class_losses = torch.stack(class_losses)

        return avg_loss

    def get_class_losses(self) -> Any:
        """Get the last computed class-specific losses."""
        if hasattr(self, 'last_class_losses'):
            return self.last_class_losses.detach().cpu().numpy()
        return np.zeros(self.num_classes)

    def get_adaptive_info(self) -> Any:
        """Get current adaptive parameters for logging."""
        kappa_tensor = getattr(self, 'kappa_values')
        return {
            'kappa_values': kappa_tensor.detach().cpu().numpy().tolist(),
            'lambda_val': self.lambda_val,
            'num_classes': self.num_classes
        }


class DynamicWeightedLoss(nn.Module):
    """
    Dynamic weighted loss that adapts based on class performance.
    Useful for handling class imbalance in medical segmentation.
    """

    def __init__(self, num_classes: int = 4, initial_weights=None):
        """Private init function."""
        super().__init__()
        self.num_classes = num_classes

        if initial_weights is not None:
            self.register_buffer(
                'class_weights', torch.tensor(
                    initial_weights, dtype=torch.float32))
        else:
            self.register_buffer('class_weights', torch.ones(num_classes))

    def update_weights(self, class_accuracies) -> None:
        """Update weights based on class performance (inverse relationship)."""
        if isinstance(class_accuracies, (list, np.ndarray)):
            class_accuracies = torch.tensor(class_accuracies, dtype=torch.float32)

        # Inverse weighting: lower accuracy = higher weight
        epsilon = 1e-6
        inverse_weights = 1.0 / (class_accuracies + epsilon)
        normalized_weights = inverse_weights / torch.sum(inverse_weights)

        device = next(self.parameters()).device
        self.class_weights.data = normalized_weights.to(device)

    def forward(self, loss_per_class):
        """Apply dynamic weighting to class-specific losses."""
        if isinstance(loss_per_class, (list, np.ndarray)):
            loss_per_class = torch.tensor(loss_per_class, dtype=torch.float32)

        weights_tensor = getattr(self, 'class_weights')
        weighted_loss = torch.sum(loss_per_class * weights_tensor)
        return weighted_loss
