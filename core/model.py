"""
LoopxDINO Model
Integrates Perception Backbone (System 1) and Looped Reasoning Decoder (System 2).
Provides seamless API for fixed-step reasoning and dynamic Shannon Entropy ACT.
"""

from typing import Any

import torch
from torch import nn

from core.act import PonderTracker, ShannonEntropyHalting
from core.backbone import DINOBackbone
from core.decoder import LoopedCrossAttentionDecoder
from core.heads import ClassificationHead, DenseMaskHead


class LoopxDINO(nn.Module):
    """
    LoopxDINO:
    Iterative System-2 Reasoning Architecture for Dense Vision on 16GB Consumer GPUs.

    Args:
        backbone: Model identifier (e.g. 'facebook/dinov3-vitb16-pretrain-lvd1689m').
        num_loops: Default reasoning iterations in fixed mode.
        d_model: Hidden dimension for reasoning decoder (matches backbone embed_dim).
        nhead: Number of attention heads in looped decoder.
        dim_feedforward: Hidden dimension of FFN in looped decoder.
        max_loops: Upper bound on reasoning loops.
        use_checkpoint: Enforce torch.utils.checkpoint in looped decoder.
        use_self_attn: Enable query-to-query self-attention inside the loop.
        dynamic_halt: If True, uses Shannon Entropy Halting (ACT) during inference.
        entropy_threshold: Epsilon threshold for early exit in ACT.
        force_mock_backbone: If True, uses structural mock ViT without contacting HuggingFace.
        task_head: Optional task head ('mask', 'classification', or None).
        num_classes: Number of classes if task_head is enabled.
    """

    def __init__(
        self,
        backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_loops: int = 4,
        d_model: int | None = None,
        nhead: int = 12,
        dim_feedforward: int = 2048,
        max_loops: int = 16,
        use_checkpoint: bool = True,
        use_self_attn: bool = True,
        use_step_embed: bool = True,
        dynamic_halt: bool = False,
        entropy_threshold: float = 0.20,
        force_mock_backbone: bool = False,
        task_head: str | None = None,
        num_classes: int | None = None,
        num_queries: int | None = None,
        step_mode: str = "adaln",
    ):
        super().__init__()
        self.num_loops = num_loops
        self.dynamic_halt = dynamic_halt
        self.entropy_threshold = entropy_threshold
        self.num_classes = num_classes

        # 1. System 1: Frozen Perception Backbone
        self.backbone = DINOBackbone(
            model_name=backbone,
            embed_dim=d_model,
            force_mock=force_mock_backbone,
        )
        self.d_model = self.backbone.embed_dim

        # 2. System 2: Looped Cross-Attention Decoder (<10M params)
        self.decoder = LoopedCrossAttentionDecoder(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            max_loops=max_loops,
            use_self_attn=use_self_attn,
            use_checkpoint=use_checkpoint,
            use_step_embed=use_step_embed,
            step_mode=step_mode,
        )

        # 3. Optional ACT Halting Monitor
        self.halting_monitor = ShannonEntropyHalting(
            entropy_threshold=entropy_threshold,
            min_loops=1,
            max_loops=max_loops,
        )
        self.ponder_tracker = PonderTracker(max_loops=max_loops)

        # 4. Optional Task Heads
        self.task_head_type = task_head
        if task_head == "mask":
            self.task_head = DenseMaskHead(
                d_model=self.d_model,
                num_classes=num_classes,
                patch_size=self.backbone.patch_size,
            )
        elif task_head == "classification":
            self.task_head = ClassificationHead(
                d_model=self.d_model,
                num_classes=num_classes or 1000,
            )
        else:
            self.task_head = None

        # 5. Persistent Learned Query Bank
        # Anchors semantic/instance slots across reasoning iterations instead of white noise.
        q_count = num_queries or num_classes
        if q_count is not None:
            self.learned_query_bank = nn.Parameter(
                torch.randn(1, q_count, self.d_model) * 0.02
            )
        else:
            self.learned_query_bank = None

    def train(self, mode: bool = True):
        """
        Enforce that backbone stays strictly frozen in eval mode even when model.train() is called.
        """
        super().train(mode)
        self.backbone.train(False)
        return self

    def extract_visual_features(
        self,
        image: torch.Tensor,
        return_grid: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[int, int]]:
        """Runs frozen System 1 visual perception stream."""
        return self.backbone(image, return_patch_grid=return_grid)

    def forward(
        self,
        image: torch.Tensor,
        queries: torch.Tensor | None = None,
        num_loops: int | None = None,
        return_trajectory: bool = False,
        return_attn_weights: bool = False,
        dynamic_halt: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any] | dict[str, Any]:
        """
        End-to-end forward pass:
          Image -> System 1 (Frozen Backbone) -> Visual Memory (KV)
          Queries + Visual Memory -> System 2 (Looped Reasoning) -> Refined Queries

        Args:
            image: High-resolution image tensor of shape (B, 3, H, W).
            queries: Initial query tokens of shape (B, L_q, d_model).
                     If None, uses the internal learned_query_bank.
            num_loops: Overrides default reasoning loops.
            return_trajectory: Returns intermediate query states across loops.
            return_attn_weights: Returns cross-attention heatmaps for visualization.
            dynamic_halt: Overrides dynamic ACT halting flag.

        Returns:
            refined_features: Tensor of shape (B, L_q, d_model), or dict if extra outputs requested.
        """
        loops = num_loops or self.num_loops
        use_dynamic_halt = self.dynamic_halt if dynamic_halt is None else dynamic_halt

        # Resolve queries: use passed queries or expand learned query bank
        if queries is None:
            if self.learned_query_bank is not None:
                queries = self.learned_query_bank.expand(image.shape[0], -1, -1)
            else:
                raise ValueError(
                    "queries must be provided when model has no internal learned_query_bank."
                )

        # 1. System 1: Extract visual memory (no gradient computed for backbone)
        visual_memory, grid_size = self.backbone(image, return_patch_grid=True)

        # 2. Dynamic ACT Inference (Early Exit via Shannon Entropy)
        if use_dynamic_halt and not self.training and self.task_head is not None:
            q_final, traj, attn_maps, actual_loops, mean_entropy = (
                self._forward_dynamic_act(
                    queries=queries,
                    visual_memory=visual_memory,
                    max_loops=loops,
                    return_attn_weights=return_attn_weights,
                )
            )
            self.ponder_tracker.record(
                actual_loops, mean_entropy, batch_size=image.shape[0]
            )

            if self.task_head_type == "mask":
                out_pred = self.task_head(
                    q_final,
                    visual_memory,
                    grid_size,
                    out_size=(image.shape[2], image.shape[3]),
                )
            elif self.task_head_type == "classification":
                out_pred = self.task_head(q_final)
            else:
                out_pred = q_final

            if return_trajectory or return_attn_weights:
                return {
                    "output": out_pred,
                    "refined_features": q_final,
                    "trajectory": traj,
                    "attn_weights": attn_maps,
                    "grid_size": grid_size,
                    "ponder_steps": actual_loops,
                    "entropy": mean_entropy,
                }
            return out_pred

        # 3. Standard Fixed-loop Forward Pass (System 2)
        decoder_output = self.decoder(
            queries=queries,
            visual_memory=visual_memory,
            num_loops=loops,
            return_trajectory=return_trajectory,
            return_attn_weights=return_attn_weights,
        )

        if return_trajectory and return_attn_weights:
            refined_features, trajectory, attn_maps = decoder_output
        elif return_trajectory:
            refined_features, trajectory = decoder_output
            attn_maps = None
        elif return_attn_weights:
            refined_features, attn_maps = decoder_output
            trajectory = None
        else:
            refined_features = decoder_output
            trajectory = None
            attn_maps = None

        # Apply Task Head if present
        if self.task_head is not None:
            if self.task_head_type == "mask":
                out_pred = self.task_head(
                    refined_features,
                    visual_memory,
                    grid_size,
                    out_size=(image.shape[2], image.shape[3]),
                )
            elif self.task_head_type == "classification":
                out_pred = self.task_head(refined_features)
            else:
                out_pred = refined_features
        else:
            out_pred = refined_features

        if return_trajectory or return_attn_weights:
            return {
                "output": out_pred,
                "refined_features": refined_features,
                "trajectory": trajectory,
                "attn_weights": attn_maps,
                "grid_size": grid_size,
            }

        return out_pred

    def _forward_dynamic_act(
        self,
        queries: torch.Tensor,
        visual_memory: torch.Tensor,
        max_loops: int,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], int, float]:
        """
        Executes dynamic ACT early exit loop during inference.
        """
        k_cache, v_cache = self.decoder.precompute_kv(visual_memory)
        q = queries
        traj = [q]
        attn_maps = []
        actual_loops = 0
        final_entropy = 1.0

        for step in range(max_loops):
            actual_loops = step + 1
            step_tensor = torch.tensor([step], device=queries.device, dtype=torch.long)

            if return_attn_weights:
                q, attn_w = self.decoder._step_forward(
                    q, k_cache, v_cache, step_tensor, return_attn_weights=True
                )
                attn_maps.append(attn_w)
            else:
                q, _ = self.decoder._step_forward(
                    q, k_cache, v_cache, step_tensor, return_attn_weights=False
                )

            traj.append(q)

            # Check halting condition using current prediction entropy
            q_norm = self.decoder.final_norm(q)
            if self.task_head_type == "classification":
                step_logits = self.task_head(q_norm)
            elif (
                self.task_head_type == "mask" and self.task_head.class_head is not None
            ):
                step_logits = self.task_head.class_head(q_norm)
            else:
                step_logits = q_norm

            halt, _, final_entropy = self.halting_monitor.should_halt(step_logits, step)
            if halt:
                break

        q_final = self.decoder.final_norm(q)
        return q_final, traj, attn_maps, actual_loops, final_entropy
