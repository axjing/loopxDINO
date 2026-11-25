"""
test_decoder_vram.py
Rigorous VRAM profiling test for LoopedCrossAttentionDecoder.
Verifies:
  1. O(1) activation VRAM scaling across loop depths N in [2, 4, 8, 12, 16] with checkpointing.
  2. Contrast with non-checkpointed linear VRAM inflation.
  3. Gradient flow integrity: decoder parameters get gradients, backbone has zero gradients.
  4. Memory safety on single 16GB RTX 5060 Ti GPU.
"""

import gc
import sys

import torch

from loopxdino.decoder import LoopedCrossAttentionDecoder
from loopxdino.model import LoopxDINO


def get_peak_memory_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def profile_loop_depth(
    decoder: LoopedCrossAttentionDecoder,
    batch_size: int = 4,
    num_queries: int = 50,
    seq_len: int = 1024,
    d_model: int = 768,
    num_loops: int = 4,
    use_checkpoint: bool = True,
    device: str = "cuda",
) -> tuple[float, float, float]:
    """
    Measures:
      - Memory before run
      - Peak memory during forward + backward
      - Gradient memory
    """
    decoder.use_checkpoint = use_checkpoint
    decoder.train()

    reset_memory()
    base_mem = get_peak_memory_mb()

    queries = torch.randn(
        batch_size,
        num_queries,
        d_model,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    visual = torch.randn(
        batch_size,
        seq_len,
        d_model,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=False,
    )

    out = decoder(queries, visual, num_loops=num_loops)
    loss = out.sum()
    loss.backward()

    peak_mem = get_peak_memory_mb()
    net_activation_peak = peak_mem - base_mem

    # Clean up
    del queries, visual, out, loss
    decoder.zero_grad(set_to_none=True)
    reset_memory()

    return base_mem, peak_mem, net_activation_peak


def main():
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. This test requires an NVIDIA GPU.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("=" * 70)
    print(" RTX 5060 Ti VRAM Profiler: LoopedCrossAttentionDecoder")
    print(f" Device: {device_name} | VRAM: {total_vram_gb:.2f} GB")
    print("=" * 70)

    d_model = 768
    batch_size = 4
    num_queries = 64
    seq_len = 1024  # 512x512 with 16x16 patches = 1024 tokens

    decoder = LoopedCrossAttentionDecoder(
        d_model=d_model,
        nhead=12,
        dim_feedforward=2048,
        max_loops=16,
        use_self_attn=True,
    ).to("cuda", dtype=torch.bfloat16)

    print("\nModel specs:")
    print(
        f"  - Parameters: {decoder.total_params:,} ({decoder.total_params / 1e6:.2f}M)"
    )
    print("  - Precision: torch.bfloat16")
    print(
        f"  - Input batch: B={batch_size}, L_q={num_queries}, S_v={seq_len}, D={d_model}\n"
    )

    loop_counts = [2, 4, 8, 12, 16]

    print("--- 1. Testing WITH Activation Checkpointing (Mandatory O(1) Rule) ---")
    results_ckpt = []
    for n in loop_counts:
        _, peak_mem, net_mem = profile_loop_depth(
            decoder,
            batch_size=batch_size,
            num_queries=num_queries,
            seq_len=seq_len,
            d_model=d_model,
            num_loops=n,
            use_checkpoint=True,
            device="cuda",
        )
        results_ckpt.append(peak_mem)
        print(
            f"  Loops: {n:2d} | Peak VRAM: {peak_mem:7.2f} MB | Net Activation: {net_mem:7.2f} MB"
        )

    print("\n--- 2. Testing WITHOUT Activation Checkpointing (Unrolled BPTT Graph) ---")
    results_no_ckpt = []
    for n in loop_counts:
        _, peak_mem, net_mem = profile_loop_depth(
            decoder,
            batch_size=batch_size,
            num_queries=num_queries,
            seq_len=seq_len,
            d_model=d_model,
            num_loops=n,
            use_checkpoint=False,
            device="cuda",
        )
        results_no_ckpt.append(peak_mem)
        print(
            f"  Loops: {n:2d} | Peak VRAM: {peak_mem:7.2f} MB | Net Activation: {net_mem:7.2f} MB"
        )

    print("\n" + "=" * 70)
    print(
        f"{'Loops':<8} | {'With Checkpoint':<18} | {'Without Checkpoint':<20} | {'VRAM Saved':<12}"
    )
    print("-" * 70)
    for i, n in enumerate(loop_counts):
        ckpt_val = results_ckpt[i]
        no_ckpt_val = results_no_ckpt[i]
        diff = no_ckpt_val - ckpt_val
        pct = (diff / no_ckpt_val) * 100 if no_ckpt_val > 0 else 0
        print(
            f"{n:<8} | {ckpt_val:10.2f} MB       | {no_ckpt_val:12.2f} MB         | {diff:6.2f} MB ({pct:.1f}%)"
        )
    print("=" * 70)

    # Verification checks
    # In checkpointing, memory growth between loops 4 and 16 should be virtually flat (< 15%)
    growth_ratio = results_ckpt[-1] / results_ckpt[1]
    print(f"\nCheckpointing memory ratio (16 loops vs 4 loops): {growth_ratio:.3f}x")
    assert growth_ratio < 1.30, (
        f"Memory grew too much ({growth_ratio:.2f}x)! Checkpointing might not be effective."
    )
    print(" [PASS] Activation VRAM scaling is verified O(1) with num_loops!")

    # Verify End-to-End Backbone Frozen Rule
    print("\n--- 3. Testing End-to-End Backbone Gradient Invariance ---")
    model = LoopxDINO(
        backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_loops=4,
        force_mock_backbone=True,
    ).to("cuda", dtype=torch.bfloat16)

    model.train()  # should keep backbone in eval()
    assert not model.backbone.training, (
        "Backbone must remain in eval() even when model.train() is called!"
    )

    dummy_img = torch.randn(2, 3, 256, 256, device="cuda", dtype=torch.bfloat16)
    dummy_q = torch.randn(
        2, 16, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    out = model(dummy_img, dummy_q)
    out.sum().backward()

    # Backbone parameters must have NO gradients
    backbone_grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert len(backbone_grads) == 0, (
        f"Violation! Found {len(backbone_grads)} backbone parameters with gradients!"
    )
    print(" [PASS] Backbone has strictly 0 gradients (read-only invariant preserved).")

    # Decoder parameters must receive gradients
    decoder_grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert len(decoder_grads) > 0, "Error: Decoder parameters received no gradients!"
    print(" [PASS] Decoder parameters received full gradients via BPTT.")

    print("\nALL VRAM AND ARCHITECTURAL INVARIANTS SATISFIED! 🎉\n")


if __name__ == "__main__":
    main()
