"""
Loss functions for loopxDINO dense semantic segmentation.
Implements:
  1. MultiClassDiceLoss: Soft multiclass Dice loss supporting ignore_index (Pascal VOC / COCO).
  2. CombinedCEDiceLoss: Hybrid Cross-Entropy and Dice loss for joint pixel-level and IoU-level optimization.
"""

import torch
import torch.nn.functional as F
from torch import nn


class MultiClassDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss.
    Optimizes intersection-over-union directly, mitigating class imbalance and enhancing boundary sharpness.

    Args:
        num_classes: Total number of semantic classes (e.g. 21 for Pascal VOC).
        ignore_index: Label value to mask out during loss computation (e.g. 255).
        smooth: Smoothing factor to prevent zero-division and stabilize gradients.
        eps: Small epsilon for numerical stability.
    """

    def __init__(
        self,
        num_classes: int = 21,
        ignore_index: int = 255,
        smooth: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits of shape (B, C, H, W).
            targets: Ground-truth class labels of shape (B, H, W).

        Returns:
            dice_loss: Scalar loss tensor.
        """
        B, C, H, W = logits.shape

        # 1. Softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # 2. Filter out ignore_index pixels (boundary void regions in VOC)
        valid_mask = (targets != self.ignore_index)  # (B, H, W)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        clean_targets = targets.clone()
        clean_targets[~valid_mask] = 0  # safe indexing for one-hot

        # 3. One-hot encoding of targets: (B, C, H, W)
        target_one_hot = torch.zeros(
            (B, C, H, W), device=logits.device, dtype=logits.dtype
        )
        target_one_hot.scatter_(1, clean_targets.unsqueeze(1), 1.0)

        # Apply valid mask across all channels
        valid_mask_expanded = valid_mask.unsqueeze(1).expand(-1, C, -1, -1)
        probs_valid = probs * valid_mask_expanded
        target_valid = target_one_hot * valid_mask_expanded

        # 4. Compute Dice per class: sum across (B, H, W)
        intersection = (probs_valid * target_valid).sum(dim=(0, 2, 3))
        cardinality = (probs_valid.pow(2) + target_valid.pow(2)).sum(dim=(0, 2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)

        # Average across classes that actually appear in the batch + background
        dice_loss = 1.0 - dice_score.mean()
        return dice_loss


class CombinedCEDiceLoss(nn.Module):
    """
    Hybrid Cross-Entropy and Soft Dice Loss.
    Combines pixel-wise cross-entropy with region-level Dice loss.

    Formula:
        L_total = ce_weight * L_CE + dice_weight * L_Dice
    """

    def __init__(
        self,
        num_classes: int = 21,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = MultiClassDiceLoss(
            num_classes=num_classes, ignore_index=ignore_index
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if self.ce_weight > 0.0:
            loss = loss + self.ce_weight * self.ce_loss(logits, targets)
        if self.dice_weight > 0.0:
            loss = loss + self.dice_weight * self.dice_loss(logits, targets)
        return loss
