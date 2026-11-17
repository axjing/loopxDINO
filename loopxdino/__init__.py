"""
loopxDINO: Iterative System-2 Reasoning for Dense Vision on 16GB Consumer GPUs.
"""

from core.act import MultiStepCascadeLoss, PonderTracker, ShannonEntropyHalting
from core.backbone import DINOBackbone
from core.decoder import LoopedCrossAttentionDecoder
from core.heads import ClassificationHead, DenseMaskHead
from core.model import LoopxDINO

__all__ = [
    "ClassificationHead",
    "DINOBackbone",
    "DenseMaskHead",
    "LoopedCrossAttentionDecoder",
    "LoopxDINO",
    "MultiStepCascadeLoss",
    "PonderTracker",
    "ShannonEntropyHalting",
]
