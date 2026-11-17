"""
Core modules for loopxDINO:
  - DINOBackbone (System 1)
  - LoopedCrossAttentionDecoder (System 2)
  - LoopxDINO (Unified Model)
  - ACT (Shannon Entropy Halting & Cascade Loss)
  - Heads (DenseMaskHead, ClassificationHead)
"""

from core.act import (
    MultiStepCascadeLoss,
    PonderTracker,
    ShannonEntropyHalting,
    compute_shannon_entropy,
)
from core.backbone import DINOBackbone, MockDINOViT
from core.decoder import LoopedCrossAttentionDecoder
from core.heads import ClassificationHead, DenseMaskHead
from core.model import LoopxDINO

__all__ = [
    "ClassificationHead",
    "DINOBackbone",
    "DenseMaskHead",
    "LoopedCrossAttentionDecoder",
    "LoopxDINO",
    "MockDINOViT",
    "MultiStepCascadeLoss",
    "PonderTracker",
    "ShannonEntropyHalting",
    "compute_shannon_entropy",
]
