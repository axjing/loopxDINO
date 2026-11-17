"""
visualize_attention.py
Visualizes Attention Evolution Maps across reasoning loops (System 2 Iterations).
Validates Section 5.3 in method.md:
  Demonstrates how cross-attention transitions from diffuse/blurry to sharp and edge-focused.
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from core.model import LoopxDINO


def compute_attention_metrics(attn_maps: list[torch.Tensor]) -> list[dict]:
    """
    Computes dispersion entropy and peak concentration across loops.
    Lower entropy and higher peak concentration indicate sharper attention focus.
    """
    metrics = []
    for step, attn in enumerate(attn_maps):
        # attn shape: (B, L_q, S_v)
        # Select first sample and first query token
        a = attn[0, 0].float()  # (S_v,)
        a = a / (a.sum() + 1e-9)

        entropy = -(a * torch.log(a + 1e-12)).sum().item()
        peak_val = a.max().item()
        top5_pct = a.topk(k=max(1, int(len(a) * 0.05)))[0].sum().item() * 100.0

        metrics.append(
            {
                "step": step + 1,
                "entropy": round(entropy, 3),
                "peak_val": round(peak_val, 4),
                "top5_pct_energy": round(top5_pct, 1),
            }
        )
    return metrics


def render_ascii_heatmap(
    matrix: torch.Tensor, width: int = 16, height: int = 16
) -> str:
    """
    Renders 2D attention tensor as a compact terminal ASCII heatmap.
    """
    chars = " .:-=+*#%@"
    mat_2d = F.interpolate(
        matrix.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear"
    ).squeeze()
    mat_norm = (mat_2d - mat_2d.min()) / (mat_2d.max() - mat_2d.min() + 1e-8)

    lines = []
    for row in range(height):
        line = "".join(chars[int(val * (len(chars) - 1))] for val in mat_norm[row])
        lines.append("    " + line)
    return "\n".join(lines)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(" loopxDINO Attention Evolution Map Generator")
    print(" Validating iterative System-2 edge purification (method.md Sec 5.3)")
    print("=" * 65)

    num_loops = 4
    model = LoopxDINO(
        backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_loops=num_loops,
        force_mock_backbone=True,
    ).to(device, dtype=torch.bfloat16)
    model.eval()

    img_size = 256
    image = torch.randn(1, 3, img_size, img_size, device=device, dtype=torch.bfloat16)
    queries = torch.randn(1, 4, 768, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = model(
            image,
            queries,
            return_attn_weights=True,
            return_trajectory=True,
        )

    attn_maps = out["attn_weights"]  # List of 4 tensors, each (1, 4, S_v)
    grid_size = out["grid_size"]  # (Hp, Wp)
    Hp, Wp = grid_size
    print(f"\nSpatial feature grid: {Hp}x{Wp} ({Hp * Wp} tokens)")
    print(f"Number of reasoning loops: {len(attn_maps)}\n")

    metrics = compute_attention_metrics(attn_maps)
    print(
        f"{'Loop Step':<12} | {'Entropy (Dispersion)':<22} | {'Peak Value':<14} | {'Top 5% Energy':<16}"
    )
    print("-" * 68)
    for m in metrics:
        print(
            f"Step {m['step']:<7} | {m['entropy']:<22} | {m['peak_val']:<14} | {m['top5_pct_energy']}%"
        )

    print("\n--- Terminal ASCII Attention Evolution (Query #0) ---")
    for step_idx, attn in enumerate(attn_maps):
        map_2d = attn[0, 0].view(Hp, Wp).float().cpu()
        print(f"\n[Loop Step {step_idx + 1}] Attention Heatmap:")
        print(render_ascii_heatmap(map_2d, width=16, height=8))

    # Save PNG visualization figure
    os.makedirs("assets", exist_ok=True)
    save_path = "assets/attention_evolution_maps.png"

    fig, axes = plt.subplots(1, num_loops, figsize=(4 * num_loops, 4))
    if num_loops == 1:
        axes = [axes]

    for step, ax in enumerate(axes):
        heat_2d = attn_maps[step][0, 0].view(Hp, Wp).float().cpu().numpy()
        im = ax.imshow(heat_2d, cmap="inferno", interpolation="bicubic")
        ax.set_title(
            f"Loop {step + 1}\nEntropy: {metrics[step]['entropy']}", fontsize=12
        )
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "loopxDINO Attention Evolution: Diffuse Perception -> Edge-Purified Focus",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAttention Evolution Map figure saved to: {os.path.abspath(save_path)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
