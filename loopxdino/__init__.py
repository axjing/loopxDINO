"""
loopxDINO: Iterative System-2 Reasoning for Dense Vision on 16GB Consumer GPUs.
"""

from loopxdino.act import (
    MultiStepCascadeLoss,
    PonderTracker,
    ShannonEntropyHalting,
    compute_shannon_entropy,
)
from loopxdino.backbone import DINOBackbone, MockDINOViT
from loopxdino.decoder import LoopedCrossAttentionDecoder
from loopxdino.heads import ClassificationHead, DenseMaskHead
from loopxdino.model import LoopxDINO

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
