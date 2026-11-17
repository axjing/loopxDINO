"""
Baseline Architectures for loopxDINO Benchmarking.
Implements:
  1. BaselineA_Linear: Direct perception projection (System 1 Only, ~0.6M params).
  2. BaselineB_TransformerDecoder: 2-layer standard unlooped Transformer Decoder (~15.7M params).
  3. BaselineC_TransformerDecoder: 6-layer standard unlooped Transformer Decoder (~47.2M params).
All baselines use the identical frozen DINO perception backbone to ensure 100% fair comparison.
"""

import torch
import torch.nn.functional as F
from torch import nn

from core.backbone import DINOBackbone
from core.heads import DenseMaskHead


class BaselineA_Linear(nn.Module):
    """
    Baseline-A: System 1 Only (Direct Linear/Convolutional Projection).
    Directly predicts 21-class segmentation masks from frozen DINO features without System-2 reasoning.
    """

    def __init__(
        self,
        backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_classes: int = 21,
        force_mock_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = DINOBackbone(
            model_name=backbone, force_mock=force_mock_backbone
        )
        self.d_model = self.backbone.embed_dim
        self.num_classes = num_classes

        # Progressive upsampling projection head (~0.6M params)
        self.head = nn.Sequential(
            nn.Conv2d(self.d_model, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    @property
    def decoder_params(self) -> int:
        return sum(p.numel() for p in self.head.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(False)
        return self

    def forward(
        self,
        image: torch.Tensor,
        queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _C, H, W = image.shape
        visual_memory, (Hp, Wp) = self.backbone(image, return_patch_grid=True)
        # Reshape to (B, D, Hp, Wp)
        vis_2d = visual_memory.transpose(1, 2).view(B, self.d_model, Hp, Wp)
        mask_logits = self.head(vis_2d)  # (B, num_classes, H_up, W_up)
        if mask_logits.shape[-2:] != (H, W):
            mask_logits = F.interpolate(
                mask_logits, size=(H, W), mode="bilinear", align_corners=False
            )
        return mask_logits


class StandardTransformerDecoder(nn.Module):
    """
    Standard non-looped, unshared multi-layer Transformer Decoder without checkpointing.
    Used for Baseline-B (2 layers) and Baseline-C (6 layers).
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, queries: torch.Tensor, visual_memory: torch.Tensor
    ) -> torch.Tensor:
        out = self.layers(tgt=queries, memory=visual_memory)
        return self.norm(out)


class BaselineTransformerBase(nn.Module):
    """Base wrapper for unlooped Transformer Decoders with identical mask head."""

    def __init__(
        self,
        num_layers: int,
        backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_classes: int = 21,
        force_mock_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = DINOBackbone(
            model_name=backbone, force_mock=force_mock_backbone
        )
        self.d_model = self.backbone.embed_dim
        self.num_classes = num_classes

        self.decoder = StandardTransformerDecoder(
            d_model=self.d_model,
            nhead=12,
            num_layers=num_layers,
            dim_feedforward=2048,
        )
        self.mask_head = DenseMaskHead(
            d_model=self.d_model,
            num_classes=num_classes,
            patch_size=self.backbone.patch_size,
        )
        self.learned_query_bank = nn.Parameter(
            torch.randn(1, num_classes, self.d_model) * 0.02
        )

    @property
    def decoder_params(self) -> int:
        return sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        ) + sum(p.numel() for p in self.mask_head.parameters() if p.requires_grad) + self.learned_query_bank.numel()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(False)
        return self

    def forward(
        self,
        image: torch.Tensor,
        queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _B, _C, H, W = image.shape
        if queries is None:
            queries = self.learned_query_bank.expand(image.shape[0], -1, -1)
        visual_memory, grid_size = self.backbone(image, return_patch_grid=True)
        # Standard unshared multi-layer decoder forward
        q_refined = self.decoder(queries, visual_memory)
        # Project through dense mask head: (B, 21, H, W)
        mask_logits, _ = self.mask_head(
            q_refined, visual_memory, grid_size, out_size=(H, W)
        )
        return mask_logits


class BaselineB_TransformerDecoder(BaselineTransformerBase):
    """Baseline-B: 2-layer standard Transformer Decoder (~15.7M params)."""

    def __init__(
        self,
        backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_classes: int = 21,
        force_mock_backbone: bool = False,
    ):
        super().__init__(
            num_layers=2,
            backbone=backbone,
            num_classes=num_classes,
            force_mock_backbone=force_mock_backbone,
        )


class BaselineC_TransformerDecoder(BaselineTransformerBase):
    """Baseline-C: 6-layer standard deep Transformer Decoder (~47.2M params)."""

    def __init__(
        self,
        backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_classes: int = 21,
        force_mock_backbone: bool = False,
    ):
        super().__init__(
            num_layers=6,
            backbone=backbone,
            num_classes=num_classes,
            force_mock_backbone=force_mock_backbone,
        )
