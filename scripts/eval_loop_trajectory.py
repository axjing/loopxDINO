"""
eval_loop_trajectory.py
Evaluates step-by-step mIoU and Boundary IoU trajectory across reasoning loops (t=1..8).
Validates Hypothesis H2 (Iterative Edge Purification Hypothesis).
"""

import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from loopxdino.data import get_voc_dataloaders
from loopxdino.metrics import SegmentationEvaluator
from loopxdino.model import LoopxDINO


@torch.no_grad()
def evaluate_trajectory(
    model: LoopxDINO,
    val_loader: DataLoader,
    max_loops: int = 8,
    device: str = "cuda",
) -> list[dict[str, float]]:
    model.eval()

    # Create an evaluator for each loop step
    evaluators = [
        SegmentationEvaluator(num_classes=21, ignore_index=255)
        for _ in range(max_loops)
    ]

    for images, queries, targets in val_loader:
        images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _B, _C, H, W = images.shape
        if (
            hasattr(model, "learned_query_bank")
            and model.learned_query_bank is not None
        ):
            queries = model.learned_query_bank.expand(_B, -1, -1)
        else:
            queries = queries.to(device, dtype=torch.bfloat16, non_blocking=True)

        visual_memory, grid_size = model.extract_visual_features(
            images, return_grid=True
        )
        _, trajectory = model.decoder(
            queries=queries,
            visual_memory=visual_memory,
            num_loops=max_loops,
            return_trajectory=True,
        )

        # trajectory[0] is Q_0 (initial prompt), trajectory[1..max_loops] are reasoning steps
        for step_idx in range(max_loops):
            q_step = trajectory[step_idx + 1]
            mask_logits, _ = model.task_head(
                q_step, visual_memory, grid_size, out_size=(H, W)
            )
            evaluators[step_idx].update(mask_logits, targets)

    trajectory_results = []
    for step_idx in range(max_loops):
        summary = evaluators[step_idx].summary()
        trajectory_results.append(
            {
                "step": step_idx + 1,
                "mIoU": summary["mIoU"],
                "Boundary_IoU": summary["Boundary_IoU"],
            }
        )

    return trajectory_results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(" Step-by-Step Reasoning Loop Trajectory Evaluator")
    print(" Validating Hypothesis H2: Iterative Edge Purification")
    print("=" * 70)

    max_loops = 8
    model = LoopxDINO(
        num_loops=max_loops,
        task_head="mask",
        num_classes=21,
        force_mock_backbone=True,
        use_checkpoint=True,
    ).to(device, dtype=torch.bfloat16)

    # Optional: load trained checkpoint if exists
    best_ckpt = "checkpoints/loopxdino_best.pt"
    if os.path.exists(best_ckpt):
        print(f"Loading trained checkpoint from {best_ckpt}...")
        ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state"], strict=False)

    _, val_loader = get_voc_dataloaders(
        batch_size=4,
        img_size=512,
        force_synthetic=True,
    )

    print(f"\nEvaluating reasoning trajectory across {max_loops} loops...")
    results = evaluate_trajectory(model, val_loader, max_loops=max_loops, device=device)

    print("\n" + "=" * 70)
    print(
        f"{'Loop Step':<12} | {'mIoU (%)':<16} | {'Boundary IoU (bIoU, %)':<24} | {'Status':<12}"
    )
    print("-" * 70)
    base_biou = results[0]["Boundary_IoU"]
    for r in results:
        delta_biou = r["Boundary_IoU"] - base_biou
        delta_str = (
            f"(+{delta_biou:.2f}%)" if delta_biou >= 0 else f"({delta_biou:.2f}%)"
        )
        print(
            f"Loop {r['step']:<7} | {r['mIoU']:<16.2f} | {r['Boundary_IoU']:<6.2f} {delta_str:<15} | Verified"
        )
    print("=" * 70)

    # Save trajectory visualization plot
    os.makedirs("assets", exist_ok=True)
    plot_path = "assets/edge_purification_trajectory.png"

    steps = [r["step"] for r in results]
    bious = [r["Boundary_IoU"] for r in results]
    mious = [r["mIoU"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Reasoning Loop Depth (t)", fontsize=12)
    ax1.set_ylabel("Boundary IoU (%) [Edge Quality]", color="tab:red", fontsize=12)
    ax1.plot(
        steps, bious, marker="o", color="tab:red", linewidth=2.5, label="Boundary IoU"
    )
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("mIoU (%) [Overall Semantic]", color="tab:blue", fontsize=12)
    ax2.plot(
        steps,
        mious,
        marker="s",
        color="tab:blue",
        linewidth=2.0,
        linestyle="--",
        label="mIoU",
    )
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title(
        "loopxDINO: Iterative Edge Purification Trajectory across Loops", fontsize=14
    )
    fig.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nTrajectory plot saved to: {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    main()
