"""
Evaluate genuine patch occlusion robustness on Pascal VOC 2012 validation set.
Runs loopxDINO (T=1, T=4, Dynamic ACT) and Baseline-A under 0%, 20%, 40%, 60% spatial patch occlusion.
Saves genuine results to experiments/real_occlusion_results.json.
"""

import json
import os

import torch
from torch.utils.data import DataLoader

from loopxdino.baselines import BaselineA_Linear, BaselineC_TransformerDecoder
from loopxdino.data import VOCSegmentationDataset
from loopxdino.metrics import SegmentationEvaluator
from loopxdino.model import LoopxDINO


def inject_patch_occlusion(images: torch.Tensor, occlusion_ratio: float) -> torch.Tensor:
    """Randomly occludes a square spatial patch covering occlusion_ratio of the total area."""
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
def evaluate_model_occlusion(
    model,
    model_type: str,
    dataloader: DataLoader,
    device: str,
    occlusion_ratios: list[float],
    num_loops: int | None = None,
    dynamic_halt: bool = False,
    max_eval_samples: int | None = 500,
) -> list[dict[str, float]]:
    model.eval()
    results = []

    for occ in occlusion_ratios:
        evaluator = SegmentationEvaluator(num_classes=21, ignore_index=255)
        count = 0
        torch.manual_seed(42)  # For reproducible occlusion patterns across models

        for images, queries, targets in dataloader:
            if max_eval_samples and count >= max_eval_samples:
                break
            images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            _B = images.shape[0]
            count += _B

            images_occ = inject_patch_occlusion(images, occ)

            if hasattr(model, "learned_query_bank") and model.learned_query_bank is not None:
                queries = model.learned_query_bank.expand(_B, -1, -1)
            else:
                queries = queries.to(device, dtype=torch.bfloat16, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if model_type == "loopxdino":
                    out = model(images_occ, queries, num_loops=num_loops, dynamic_halt=dynamic_halt)
                    if isinstance(out, tuple):
                        mask_logits = out[0]
                    elif isinstance(out, dict):
                        mask_logits = out.get("output", out.get("mask_logits", None))
                        if isinstance(mask_logits, tuple):
                            mask_logits = mask_logits[0]
                    else:
                        mask_logits = out
                else:
                    mask_logits = model(images_occ, queries)

            evaluator.update(mask_logits, targets)

        metrics = evaluator.summary()
        results.append({
            "occlusion": occ,
            "mIoU": metrics["mIoU"],
            "Boundary_IoU": metrics["Boundary_IoU"],
        })
        print(f"  [Occ {int(occ*100):2d}%] mIoU: {metrics['mIoU']:5.2f}% | bIoU: {metrics['Boundary_IoU']:5.2f}%", flush=True)

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(" Genuine Patch Occlusion Robustness Evaluation on VOC 2012 Val", flush=True)
    print("=" * 70, flush=True)

    voc_dir = os.path.join("data", "pascal_voc")
    val_dataset = VOCSegmentationDataset(
        root=voc_dir,
        image_set="val",
        img_size=512,
        download=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    print(f"Validation dataset loaded: {len(val_dataset)} images.", flush=True)

    occlusion_levels = [0.0, 0.20, 0.40, 0.60]
    out_data = {}

    # 1. Baseline-A
    print("\n--- Evaluating Baseline-A (Linear Conv) ---", flush=True)
    model_a = BaselineA_Linear(num_classes=21).to(device, dtype=torch.bfloat16)
    ckpt_a = "checkpoints/baseline_a_best.pt"
    if os.path.exists(ckpt_a):
        ckpt_dict = torch.load(ckpt_a, map_location=device, weights_only=False)
        state = ckpt_dict.get("model_state", ckpt_dict)
        model_a.load_state_dict(state, strict=False)
        print(f"Loaded weights from {ckpt_a}", flush=True)
    out_data["baseline_a"] = evaluate_model_occlusion(
        model_a, "baseline_a", val_loader, device, occlusion_levels, max_eval_samples=400
    )

    # 2. Baseline-C
    print("\n--- Evaluating Baseline-C (6L Transformer) ---", flush=True)
    model_c = BaselineC_TransformerDecoder(num_classes=21).to(device, dtype=torch.bfloat16)
    ckpt_c = "checkpoints/baseline_c_best.pt"
    if os.path.exists(ckpt_c):
        ckpt_dict = torch.load(ckpt_c, map_location=device, weights_only=False)
        state = ckpt_dict.get("model_state", ckpt_dict)
        model_c.load_state_dict(state, strict=False)
        print(f"Loaded weights from {ckpt_c}", flush=True)
    out_data["baseline_c"] = evaluate_model_occlusion(
        model_c, "baseline_c", val_loader, device, occlusion_levels, max_eval_samples=400
    )

    # 3. loopxDINO v2.0
    ckpt_loop = "checkpoints/loopxdino_best.pt"
    print("\n--- Evaluating loopxDINO v2.0 (T=1, Single-Pass) ---", flush=True)
    model_loop = LoopxDINO(
        task_head="mask",
        num_classes=21,
        num_loops=4,
        dynamic_halt=False,
    ).to(device, dtype=torch.bfloat16)
    if os.path.exists(ckpt_loop):
        ckpt_dict = torch.load(ckpt_loop, map_location=device, weights_only=False)
        state = ckpt_dict.get("model_state", ckpt_dict)
        model_loop.load_state_dict(state, strict=False)
        print(f"Loaded weights from {ckpt_loop}", flush=True)

    out_data["loopxdino_t1"] = evaluate_model_occlusion(
        model_loop, "loopxdino", val_loader, device, occlusion_levels, num_loops=1, max_eval_samples=400
    )

    print("\n--- Evaluating loopxDINO v2.0 (T=2, Optimal Step) ---", flush=True)
    out_data["loopxdino_t2"] = evaluate_model_occlusion(
        model_loop, "loopxdino", val_loader, device, occlusion_levels, num_loops=2, max_eval_samples=400
    )

    print("\n--- Evaluating loopxDINO v2.0 (T=4, Nominal Recurrence) ---", flush=True)
    out_data["loopxdino_t4"] = evaluate_model_occlusion(
        model_loop, "loopxdino", val_loader, device, occlusion_levels, num_loops=4, max_eval_samples=400
    )

    print("\n--- Evaluating loopxDINO v2.0 (Dynamic ACT Halting) ---", flush=True)
    model_loop.dynamic_halt = True
    model_loop.halting_monitor.entropy_threshold = 0.20
    out_data["loopxdino_act"] = evaluate_model_occlusion(
        model_loop, "loopxdino", val_loader, device, occlusion_levels, num_loops=8, dynamic_halt=True, max_eval_samples=400
    )

    os.makedirs("experiments", exist_ok=True)
    out_path = "experiments/real_occlusion_results.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[SUCCESS] Occlusion evaluation finished! Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
