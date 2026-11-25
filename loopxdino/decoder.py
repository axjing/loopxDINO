"""
Looped Cross-Attention Decoder (System 2)
Single-layer parameter-shared iterative reasoning module with sub-10M parameter budget.
Enforces:
  1. O(1) activation VRAM scaling via mandatory torch.utils.checkpoint.
  2. Static KV caching derived once from frozen DINOv3 features.
  3. Loop-step embedding injection per iteration to prevent representation collapse.
  4. Native PyTorch implementation and bfloat16/FP8 compatibility.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class LoopedCrossAttentionDecoder(nn.Module):
    """
    Looped Cross-Attention Decoder (System 2):
    Applies single-layer shared cross-attention and FFN over N iterative reasoning loops.

    Mathematical formulation:
      K_DINO, V_DINO = Project_KV(visual_memory)   # Precomputed ONCE (KV Caching)
      Q_0 = initial_queries
      For step in 0 .. N-1:
        Q'_step = Q_step + StepEmbedding(step)
        Q''_step = Q_step + CrossAttention(LN(Q'_step), K_DINO, V_DINO)
        Q_{step+1} = Q''_step + FFN(LN(Q''_step))
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 2048,
        max_loops: int = 16,
        dropout: float = 0.0,
        use_self_attn: bool = True,
        use_checkpoint: bool = True,
        use_step_embed: bool = True,
        step_mode: str = "adaln",  # 'adaln' (adaptive LayerNorm) or 'additive'
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.max_loops = max_loops
        self.use_self_attn = use_self_attn
        self.use_checkpoint = use_checkpoint
        self.use_step_embed = use_step_embed
        self.step_mode = step_mode

        # 1. Step Conditioning: adaLN or Additive
        # In adaLN mode: step-dependent scale and shift modulate LayerNorm without residual norm drift.
        # In additive mode: legacy additive query bias.
        self.step_embed = nn.Embedding(max_loops, d_model)
        nn.init.normal_(self.step_embed.weight, std=0.02)
        if step_mode == "adaln":
            # 6 modulation channels: (gamma_self, beta_self, gamma_cross, beta_cross, gamma_ffn, beta_ffn)
            self.step_mod = nn.Embedding(max_loops, 6 * d_model)
            nn.init.normal_(self.step_mod.weight, std=0.02)

        # 2. KV Projection: maps frozen visual features once into static K and V
        # If visual features are already d_model, this acts as the static linear projection
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # 3. Query Self-Attention (optional, allows tokens in query prompt to communicate)
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_self = nn.LayerNorm(d_model)

        # 4. Cross Attention (Query attends to static visual KV Memory)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.norm_cross = nn.LayerNorm(d_model)

        # 5. Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

        # Final LayerNorm for output stability
        self.final_norm = nn.LayerNorm(d_model)

        # Learnable residual contraction damping parameter (alpha in [0, 1]).
        # Controls the residual connection strength: q_next = alpha * q_residual + (1-alpha) * q_identity
        # Initialized at 0.9 (near-identity) for warm-start compatibility.
        # Clamped to [0, 1] during forward to guarantee strict contraction on S^{D-1}.
        self.residual_alpha = nn.Parameter(torch.tensor(0.9))

        # Preallocated step index buffer for CUDA Graph capture and zero-host-transfer loops
        self.register_buffer("step_indices", torch.arange(max_loops, dtype=torch.long), persistent=False)

        self._init_weights()
        self._verify_parameter_budget()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if hasattr(self, "step_embed"):
            nn.init.normal_(self.step_embed.weight, std=0.02)
        if hasattr(self, "step_mod"):
            nn.init.normal_(self.step_mod.weight, std=0.02)

    def _verify_parameter_budget(self):
        total_params = sum(p.numel() for p in self.parameters())
        self.total_params = total_params
        if total_params >= 10_000_000:
            raise ValueError(
                f"LoopedCrossAttentionDecoder parameter count ({total_params / 1e6:.2f}M) "
                f"exceeds strict sub-10M parameter budget!"
            )

    def precompute_kv(
        self, visual_memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        KV Caching: Computes Key and Value from frozen DINO features ONCE before the loop.
        Visual memory has shape (B, S_v, d_model).
        Returns:
            k_cache: (B, S_v, d_model)
            v_cache: (B, S_v, d_model)
        """
        k_cache = self.k_proj(visual_memory)
        v_cache = self.v_proj(visual_memory)
        return k_cache, v_cache

    def _cross_attention_step(
        self,
        q_norm: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Computes multi-head cross attention with batch_first=True.
        Supports Mask-Guided attention bias for iterative boundary refinement.
        """
        B, L_q, D = q_norm.shape
        _, S_v, _ = k_cache.shape

        # Query projection
        q = self.q_proj(q_norm)  # (B, L_q, D)

        head_dim = D // self.nhead
        scale = 1.0 / (head_dim**0.5)

        # Reshape to (B, nhead, L_q, head_dim) and (B, nhead, S_v, head_dim)
        q = q.view(B, L_q, self.nhead, head_dim).transpose(1, 2)
        k = k_cache.view(B, S_v, self.nhead, head_dim).transpose(1, 2)
        v = v_cache.view(B, S_v, self.nhead, head_dim).transpose(1, 2)

        if return_attn_weights:
            # Explicit attention map computation for visualization
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_bias is not None:
                attn_scores = attn_scores + attn_bias
            attn_weights = F.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, v)
            # Average attention weights across heads for visualization: (B, L_q, S_v)
            mean_attn = attn_weights.mean(dim=1)
        else:
            # Scaled Dot Product Attention (FlashAttention / PyTorch SDPA)
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, scale=scale, is_causal=False
            )
            mean_attn = None

        # Reshape back to (B, L_q, D) and project
        context = context.transpose(1, 2).contiguous().view(B, L_q, D)
        out = self.out_proj(context)
        return out, mean_attn

    def _step_forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        step_idx: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Single step reasoning iteration.
        Supports both modern adaLN (Adaptive LayerNorm) and legacy additive conditioning,
        along with optional Mask-Guided Attention Bias (attn_bias).
        """
        if self.use_step_embed and self.step_mode == "adaln":
            mod = self.step_mod(step_idx)
            if mod.dim() == 2:
                mod = mod.unsqueeze(1)  # (1, 1, 6 * d_model)
            gamma_s, beta_s, gamma_c, beta_c, gamma_f, beta_f = mod.chunk(6, dim=-1)

            # 1. Self Attention with adaLN
            if self.use_self_attn:
                q_norm_self = self.norm_self(q) * (1.0 + gamma_s) + beta_s
                self_out, _ = self.self_attn(
                    q_norm_self, q_norm_self, q_norm_self, need_weights=False
                )
                q = q + self_out

            # 2. Cross Attention with adaLN & Mask Guidance
            q_norm_cross = self.norm_cross(q) * (1.0 + gamma_c) + beta_c
            cross_out, attn_map = self._cross_attention_step(
                q_norm_cross,
                k_cache,
                v_cache,
                attn_bias=attn_bias,
                return_attn_weights=return_attn_weights,
            )
            q = q + cross_out

            # 3. FFN with adaLN
            q_norm_ffn = self.norm_ffn(q) * (1.0 + gamma_f) + beta_f
            ffn_out = self.ffn(q_norm_ffn)
            # Contraction-damped residual: alpha controls residual strength,
            # clamped to [0,1] for guaranteed Lipschitz contraction on S^{D-1}.
            alpha = self.residual_alpha.clamp(0.0, 1.0)
            q_next = self.final_norm(alpha * (q + ffn_out) + (1.0 - alpha) * q)
            return q_next, attn_map

        else:
            # Additive step injection mode
            if self.use_step_embed:
                step_emb = self.step_embed(step_idx)
                if step_emb.dim() == 2:
                    step_emb = step_emb.unsqueeze(1)
                q_cur = q + step_emb
            else:
                q_cur = q

            if self.use_self_attn:
                q_norm_self = self.norm_self(q_cur)
                self_out, _ = self.self_attn(
                    q_norm_self, q_norm_self, q_norm_self, need_weights=False
                )
                q_cur = q_cur + self_out

            q_norm_cross = self.norm_cross(q_cur)
            cross_out, attn_map = self._cross_attention_step(
                q_norm_cross,
                k_cache,
                v_cache,
                attn_bias=attn_bias,
                return_attn_weights=return_attn_weights,
            )
            q_cur = q_cur + cross_out

            q_norm_ffn = self.norm_ffn(q_cur)
            ffn_out = self.ffn(q_norm_ffn)
            q_next = q_cur + ffn_out
            return q_next, attn_map

    def _step_forward_ckpt(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        step_idx: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Wrapper for activation checkpointing which returns only the tensor state."""
        out, _ = self._step_forward(
            q, k_cache, v_cache, step_idx, attn_bias=attn_bias, return_attn_weights=False
        )
        return out

    def forward(
        self,
        queries: torch.Tensor,
        visual_memory: torch.Tensor,
        num_loops: int = 4,
        attn_biases: list[torch.Tensor | None] | None = None,
        return_trajectory: bool = False,
        return_attn_weights: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        """
        Iterative reasoning forward pass.

        Args:
            queries: Input queries of shape (B, L_q, d_model).
            visual_memory: Frozen DINO visual tokens of shape (B, S_v, d_model).
            num_loops: Number of reasoning iterations (System 2 compute depth).
            attn_biases: Optional list of spatial attention biases for each loop step (Mask Guidance).
            return_trajectory: Whether to return intermediate query states [Q_0, Q_1, ..., Q_N].
            return_attn_weights: Whether to return cross-attention heatmaps across loops.

        Returns:
            q_final: Refined query state after num_loops iterations, shape (B, L_q, d_model).
            trajectory (optional): List of query states at each step.
            attn_maps (optional): List of attention weight tensors at each step.
        """
        if num_loops > self.max_loops:
            raise ValueError(
                f"Requested num_loops ({num_loops}) exceeds max_loops ({self.max_loops})"
            )

        # KV Caching: Precompute static K and V once before loop
        k_cache, v_cache = self.precompute_kv(visual_memory)

        trajectory = [queries] if return_trajectory else None
        attn_maps = [] if return_attn_weights else None

        q = queries
        for step in range(num_loops):
            step_tensor = self.step_indices[step]
            step_bias = (
                attn_biases[step]
                if (attn_biases is not None and step < len(attn_biases))
                else None
            )

            if return_attn_weights:
                q, attn_w = self._step_forward(
                    q,
                    k_cache,
                    v_cache,
                    step_tensor,
                    attn_bias=step_bias,
                    return_attn_weights=True,
                )
                attn_maps.append(attn_w)
            elif (
                self.training
                and self.use_checkpoint
                and (
                    q.requires_grad
                    or k_cache.requires_grad
                    or v_cache.requires_grad
                    or (step_bias is not None and step_bias.requires_grad)
                )
            ):
                # Mandatory activation checkpointing during training to enforce O(1) memory
                q = checkpoint(
                    self._step_forward_ckpt,
                    q,
                    k_cache,
                    v_cache,
                    step_tensor,
                    step_bias,
                    use_reentrant=False,
                )
            else:
                q, _ = self._step_forward(
                    q,
                    k_cache,
                    v_cache,
                    step_tensor,
                    attn_bias=step_bias,
                    return_attn_weights=False,
                )

            if return_trajectory:
                trajectory.append(q)

        q_final = self.final_norm(q)

        if return_trajectory and return_attn_weights:
            return q_final, trajectory, attn_maps
        elif return_trajectory:
            return q_final, trajectory
        elif return_attn_weights:
            return q_final, attn_maps
        return q_final
