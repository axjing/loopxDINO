"""
benchmark_robustness_act.py
Comprehensive Evaluation of OOD / Occlusion Robustness and ACT Economic Allocation (Phase 2 & 3 / M4).
Evaluates:
  1. Patch Occlusion Resistance: Performance under 0%, 20%, 40%, 60% occlusion across loops (N=1, 4, 8).
  2. ACT Dynamic Compute Allocation: Validates Hypothesis H3 (Simple samples exit in 1-2 loops, difficult occluded samples trigger deep reasoning).
  3. Ponder Time & Compute Savings under varying complexity.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import torch

from core.data import get_voc_dataloaders
from core.metrics import SegmentationEvaluator
from core.model import LoopxDINO


def inject_patch_occlusion(
    images: torch.Tensor, occlusion_ratio: float
) -> torch.Tensor:
    """
    Randomly occludes a square spatial patch covering occlusion_ratio of the total area.
    """
    if occlusion_ratio <= 0.0:
        return images

    B, _C, H, W = images.shape
    patch_size = int((H * W * occlusion_ratio) ** 0.5)
    patch_size = min(H, max(1, patch_size))

    occluded = images.clone()
    for b in range(B):
        top = torch.randint(0, max(1, H - patch_size + 1), (1,)).item()
        left = torch.randint(0, max(1, W - patch_size + 1), (1,)).item()
        occluded[b, :, top : top + patch_size, left : left + patch_size] = 0.0

    return occluded


@torch.no_grad()
def evaluate_robustness_curve(
    model: LoopxDINO,
    val_loader,
    occlusion_ratios: list[float],
    device: str,
) -> dict[str, list[float]]:
    """Evaluates mIoU under increasing occlusion for N=1, N=4, N=8."""
    model.eval()
    results = {"Loop-1": [], "Loop-4": [], "Loop-8": []}

    for occ in occlusion_ratios:
        eval_1 = SegmentationEvaluator(num_classes=21, ignore_index=255)
        eval_4 = SegmentationEvaluator(num_classes=21, ignore_index=255)
        eval_8 = SegmentationEvaluator(num_classes=21, ignore_index=255)

        for imgs, queries, targets in val_loader:
            imgs = imgs.to(device, dtype=torch.bfloat16)
            targets = targets.to(device)
            _B = imgs.shape[0]
            if (
                hasattr(model, "learned_query_bank")
                and model.learned_query_bank is not None
            ):
                queries = model.learned_query_bank.expand(_B, -1, -1)
            else:
                queries = queries.to(device, dtype=torch.bfloat16)

            # Injected occlusion
            imgs_occ = inject_patch_occlusion(imgs, occ)

            # Loop 1
            out1 = model(imgs_occ, queries, num_loops=1)
            eval_1.update(out1, targets)

            # Loop 4
            out4 = model(imgs_occ, queries, num_loops=4)
            eval_4.update(out4, targets)

            # Loop 8
            out8 = model(imgs_occ, queries, num_loops=8)
            eval_8.update(out8, targets)

        results["Loop-1"].append(eval_1.summary()["mIoU"])
        results["Loop-4"].append(eval_4.summary()["mIoU"])
        results["Loop-8"].append(eval_8.summary()["mIoU"])

    return results


@torch.no_grad()
def evaluate_act_ponder_distribution(
    model: LoopxDINO,
    val_loader,
    occlusion_ratios: list[float],
    device: str,
) -> dict[str, Any]:
    """Measures how ACT dynamically scales its ponder time with sample difficulty."""
    model.eval()
    model.dynamic_halt = True
    model.halting_monitor.entropy_threshold = 0.30

    ponder_times = []
    savings_pcts = []

    for occ in occlusion_ratios:
        model.ponder_tracker.reset()
        for imgs, queries, targets in val_loader:
            imgs = imgs.to(device, dtype=torch.bfloat16)
            targets = targets.to(device)
            _B = imgs.shape[0]
            if (
                hasattr(model, "learned_query_bank")
                and model.learned_query_bank is not None
            ):
                queries = model.learned_query_bank.expand(_B, -1, -1)
            else:
                queries = queries.to(device, dtype=torch.bfloat16)
            imgs_occ = inject_patch_occlusion(imgs, occ)
            _ = model(imgs_occ, queries, num_loops=10, dynamic_halt=True)

        ponder_times.append(model.ponder_tracker.average_ponder_time)
        savings_pcts.append(model.ponder_tracker.compute_saving_ratio * 100.0)

    return {
        "occlusion_ratios": [f"{int(r * 100)}%" for r in occlusion_ratios],
        "average_ponder_times": ponder_times,
        "compute_savings_pct": savings_pcts,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(" M4: Occlusion Robustness & ACT Economic Allocation Benchmark")
    print(
        f" Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print("=" * 80)

    model = LoopxDINO(
        task_head="mask",
        num_classes=21,
        force_mock_backbone=True,
        dynamic_halt=True,
    ).to(device, dtype=torch.bfloat16)

    _, val_loader = get_voc_dataloaders(
        batch_size=4,
        img_size=512,
        force_synthetic=True,
    )

    occlusion_levels = [0.0, 0.20, 0.40, 0.60]

    # 1. Run Robustness Evaluation
    print("\n[Phase 2] Evaluating Occlusion Robustness vs Loop Depth...")
    rob_results = evaluate_robustness_curve(model, val_loader, occlusion_levels, device)

    # 2. Run ACT Ponder Distribution Evaluation
    print("\n[Phase 3] Evaluating ACT Ponder Time Scaling with Sample Complexity...")
    act_results = evaluate_act_ponder_distribution(
        model, val_loader, occlusion_levels, device
    )

    # ---------------------------------------------------------
    # Plot Generation
    # ---------------------------------------------------------
    os.makedirs("assets", exist_ok=True)
    plot_path = "assets/robustness_and_act_analysis.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Robustness Curve
    occ_pcts = [int(r * 100) for r in occlusion_levels]
    ax1.plot(
        occ_pcts,
        rob_results["Loop-1"],
        marker="o",
        linewidth=2.0,
        linestyle=":",
        label="Loop-1 (Shallow Perception)",
    )
    ax1.plot(
        occ_pcts,
        rob_results["Loop-4"],
        marker="s",
        linewidth=2.5,
        label="Loop-4 (Standard Reasoning)",
    )
    ax1.plot(
        occ_pcts,
        rob_results["Loop-8"],
        marker="^",
        linewidth=2.5,
        color="darkgreen",
        label="Loop-8 (Deep System-2)",
    )
    ax1.set_title("Phase 2: mIoU Retention under Severe Occlusion", fontsize=12)
    ax1.set_xlabel("Spatial Occlusion Ratio (%)", fontsize=10)
    ax1.set_ylabel("Validation mIoU (%)", fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ACT Ponder Time & Savings
    x = range(len(occlusion_levels))
    ax2.bar(
        x,
        act_results["average_ponder_times"],
        color="royalblue",
        width=0.4,
        label="Average Ponder Time (Loops)",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(act_results["occlusion_ratios"])
    ax2.set_xlabel("Scene Complexity (Occlusion Level)", fontsize=10)
    ax2.set_ylabel(
        "Average Executed Loops (Ponder Time)", color="royalblue", fontsize=10
    )
    ax2.tick_params(axis="y", labelcolor="royalblue")
    ax2.set_ylim(0, 11)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        x,
        act_results["compute_savings_pct"],
        color="crimson",
        marker="o",
        linewidth=2.5,
        label="Compute Saved (%)",
    )
    ax2_twin.set_ylabel(
        "Compute Saved vs Max (10 Loops) [%]", color="crimson", fontsize=10
    )
    ax2_twin.tick_params(axis="y", labelcolor="crimson")
    ax2_twin.set_ylim(0, 100)

    ax2.set_title(
        "Phase 3: ACT Dynamic Ponder Allocation vs Scene Complexity", fontsize=12
    )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "loopxDINO: Occlusion Robustness & ACT Economic Allocation (RTX 5060 Ti)",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # Terminal Summary Table
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(" OCCLUSION ROBUSTNESS SUMMARY (Phase 2)")
    print("=" * 80)
    print(
        f"{'Occlusion':<12} | {'Loop-1 mIoU':<14} | {'Loop-4 mIoU':<14} | {'Loop-8 mIoU':<14} | {'Loop-8 Gain':<14}"
    )
    print("-" * 75)
    for i, occ in enumerate(occ_pcts):
        l1 = rob_results["Loop-1"][i]
        l4 = rob_results["Loop-4"][i]
        l8 = rob_results["Loop-8"][i]
        gain = f"+{l8 - l1:.2f}%" if l8 >= l1 else f"{l8 - l1:.2f}%"
        print(f"{occ}%{'':<9} | {l1:<14.2f} | {l4:<14.2f} | {l8:<14.2f} | {gain:<14}")

    print("\n" + "=" * 80)
    print(" ACT ADAPTIVE COMPUTE ALLOCATION SUMMARY (Phase 3)")
    print("=" * 80)
    print(
        f"{'Complexity (Occlusion)':<25} | {'Average Ponder (Loops)':<24} | {'Compute Saved vs Max':<22}"
    )
    print("-" * 75)
    for i, occ in enumerate(act_results["occlusion_ratios"]):
        pt = act_results["average_ponder_times"][i]
        sv = act_results["compute_savings_pct"][i]
        print(f"{occ:<25} | {pt:<24.2f} | {sv:<22.1f}%")
    print("=" * 80)
    print(f"\nAnalysis plot saved to: {os.path.abspath(plot_path)}\n")


if __name__ == "__main__":
    main()
