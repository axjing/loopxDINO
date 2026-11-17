"""
Metrics Suite for Pascal VOC Dense Semantic Segmentation.
Implements:
  1. Mean Intersection-over-Union (mIoU) with ignore_index=255.
  2. Boundary IoU (bIoU) via morphological boundary bands (validating edge purification).
  3. Peak VRAM & Inference Throughput (FPS) profiler.
"""

import torch
import torch.nn.functional as F


def extract_boundary_band(
    mask: torch.Tensor,
    dilation_radius: int = 3,
) -> torch.Tensor:
    """
    Extracts object boundary band using morphological dilation and erosion.
    Boundary band B = Dilation(mask) - Erosion(mask).

    Args:
        mask: Long or float tensor of shape (B, H, W) or (B, 1, H, W).
        dilation_radius: Radius in pixels (default 3 pixels).
    Returns:
        boundary_band: Binary float tensor (B, H, W) of boundary pixels.
    """
    if mask.dim() == 3:
        m = mask.unsqueeze(1).float()
    else:
        m = mask.float()

    k_size = 2 * dilation_radius + 1
    pad = dilation_radius

    # Binary foreground indicator
    fg = (m > 0).float()

    # Morphological dilation via max_pool2d
    dilated = F.max_pool2d(fg, kernel_size=k_size, stride=1, padding=pad)
    # Morphological erosion via negated max_pool2d
    eroded = -F.max_pool2d(-fg, kernel_size=k_size, stride=1, padding=pad)

    boundary = (dilated - eroded) > 0.5
    return boundary.squeeze(1).float()


class SegmentationEvaluator:
    """
    Accumulates predictions across dataset to compute global mIoU and Boundary IoU.
    """

    def __init__(
        self,
        num_classes: int = 21,
        ignore_index: int = 255,
        boundary_radius: int = 3,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_radius = boundary_radius
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )
        self.boundary_inter = 0.0
        self.boundary_union = 0.0
        self.total_samples = 0

    def update(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
    ):
        """
        Updates evaluator state with batch predictions.

        Args:
            pred_logits: (B, num_classes, H, W) or class index tensor (B, H, W).
            target_masks: (B, H, W) ground truth labels with ignore_index=255.
        """
        if isinstance(pred_logits, tuple):
            pred_logits = pred_logits[0]
        elif isinstance(pred_logits, dict):
            pred_logits = pred_logits["output"][0] if isinstance(pred_logits["output"], tuple) else pred_logits["output"]

        if pred_logits.dim() == 4:
            preds = pred_logits.argmax(dim=1)
        else:
            preds = pred_logits

        preds_cpu = preds.detach().cpu()
        targets_cpu = target_masks.detach().cpu()

        # 1. Update Confusion Matrix for standard mIoU
        valid_mask = (
            (targets_cpu != self.ignore_index)
            & (targets_cpu >= 0)
            & (targets_cpu < self.num_classes)
        )
        t_valid = targets_cpu[valid_mask]
        p_valid = preds_cpu[valid_mask]

        indices = self.num_classes * t_valid + p_valid
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += counts.view(self.num_classes, self.num_classes)

        # 2. Update Boundary IoU
        # Extract boundary bands for ground truth and predictions
        gt_boundary = extract_boundary_band(
            targets_cpu * valid_mask.long(), dilation_radius=self.boundary_radius
        )
        pred_boundary = extract_boundary_band(
            preds_cpu * valid_mask.long(), dilation_radius=self.boundary_radius
        )

        inter = (gt_boundary * pred_boundary).sum().item()
        union = ((gt_boundary + pred_boundary) > 0.5).float().sum().item()

        self.boundary_inter += inter
        self.boundary_union += max(1.0, union)
        self.total_samples += targets_cpu.shape[0]

    def compute_miou(self) -> tuple[float, list[float]]:
        """Computes mean IoU and per-class IoU."""
        tp = torch.diag(self.confusion_matrix).float()
        pos_gt = self.confusion_matrix.sum(dim=1).float()
        pos_pred = self.confusion_matrix.sum(dim=0).float()
        denominator = pos_gt + pos_pred - tp

        ious = []
        valid_ious = []
        for c in range(self.num_classes):
            if denominator[c] > 0:
                iou = (tp[c] / denominator[c]).item()
                ious.append(iou)
                valid_ious.append(iou)
            else:
                ious.append(0.0)

        miou = sum(valid_ious) / max(1, len(valid_ious))
        return miou, ious

    def compute_boundary_iou(self) -> float:
        """Computes global Boundary IoU."""
        if self.boundary_union == 0.0:
            return 0.0
        return self.boundary_inter / self.boundary_union

    def summary(self) -> dict[str, float]:
        miou, _ = self.compute_miou()
        biou = self.compute_boundary_iou()
        return {
            "mIoU": round(miou * 100.0, 2),
            "Boundary_IoU": round(biou * 100.0, 2),
            "total_samples": self.total_samples,
        }
