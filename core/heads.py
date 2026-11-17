"""
Task Prediction Heads for loopxDINO
Implements:
  1. DenseMaskHead: Query-based dense mask prediction for segmentation (Pascal VOC / COCO).
  2. ClassificationHead: Token-level or sequence-level classification for action/scene recognition.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DenseMaskHead(nn.Module):
    """
    Query-based dense segmentation mask predictor.
    Takes refined queries (mask tokens) and visual patch features,
    upsamples spatial features, and computes dot-product masks.

      Mask_logits = einsum('bqd, bdhw -> bqhw', Query_proj, Spatial_features)
    """

    def __init__(
        self,
        d_model: int = 768,
        num_classes: int | None = None,
        feat_dim: int = 256,
        patch_size: int = 16,
        use_modulation: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        self.use_modulation = use_modulation

        # Query embedding projection
        self.query_proj = nn.Linear(d_model, feat_dim)

        # Spatial feature decoder with progressive upsampling
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(d_model, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
            nn.ConvTranspose2d(feat_dim, feat_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
            nn.ConvTranspose2d(feat_dim, feat_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
        )

        # Top-down modulation generator: System 2 Query FiLM conditioning on System 1 spatial features
        if use_modulation:
            self.spatial_mod = nn.Linear(d_model, feat_dim * 2)
            nn.init.zeros_(self.spatial_mod.weight)
            nn.init.zeros_(self.spatial_mod.bias)

        if num_classes is not None:
            self.class_head = nn.Linear(d_model, num_classes)
        else:
            self.class_head = None

    def forward(
        self,
        queries: torch.Tensor,
        visual_memory: torch.Tensor,
        grid_size: tuple[int, int],
        out_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            queries: Refined query tokens (B, L_q, d_model).
            visual_memory: Visual patch tokens (B, S_v, d_model).
            grid_size: (Hp, Wp) patch grid dimensions.
            out_size: Optional (H, W) to interpolate masks to original resolution.

        Returns:
            mask_logits: (B, L_q, H, W) mask predictions.
            class_logits: Optional (B, L_q, num_classes) category predictions.
        """
        B, _L_q, D = queries.shape
        Hp, Wp = grid_size

        # 1. Project query tokens to mask embedding vectors: (B, L_q, feat_dim)
        q_proj = self.query_proj(queries)

        # 2. Reshape visual patch tokens to 2D feature map and upsample
        # visual_memory is (B, Hp*Wp, D) -> (B, D, Hp, Wp)
        vis_2d = visual_memory.transpose(1, 2).view(B, D, Hp, Wp)
        spatial_feats = self.spatial_proj(vis_2d)  # (B, feat_dim, H_up, W_up)

        # Top-down modulation from System 2 Queries into System 1 Spatial features (FiLM)
        if self.use_modulation:
            q_global = queries.mean(dim=1)  # (B, d_model)
            mod = self.spatial_mod(q_global)  # (B, feat_dim * 2)
            scale, shift = mod.chunk(2, dim=-1)
            scale = scale.view(B, self.feat_dim, 1, 1)
            shift = shift.view(B, self.feat_dim, 1, 1)
            spatial_feats = spatial_feats * (1.0 + scale) + shift

        # 3. Dense matrix multiplication: (B, L_q, H_up, W_up)
        mask_logits = torch.einsum("bqd,bdhw->bqhw", q_proj, spatial_feats)

        if out_size is not None and mask_logits.shape[-2:] != out_size:
            mask_logits = F.interpolate(
                mask_logits, size=out_size, mode="bilinear", align_corners=False
            )

        class_logits = self.class_head(queries) if self.class_head is not None else None
        return mask_logits, class_logits


class ClassificationHead(nn.Module):
    """
    Lightweight classification head for action recognition or semantic categorization.
    Supports token-level (pool_mode='none') and sequence-level (pool_mode='mean'|'first') predictions.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_classes: int = 1000,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.pool_mode = pool_mode
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, L_q, d_model) or (B, d_model).
        Returns:
            logits: (B, num_classes) or (B, L_q, num_classes) if pool_mode == 'none'.
        """
        if queries.dim() == 3:
            if self.pool_mode == "none":
                return self.classifier(queries)
            elif self.pool_mode == "first":
                pooled = queries[:, 0]
            else:
                pooled = queries.mean(dim=1)
        else:
            pooled = queries

        return self.classifier(pooled)
