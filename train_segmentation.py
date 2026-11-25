"""
train_segmentation.py
Dedicated Pascal VOC 2012 Dense Semantic Segmentation Trainer.
Supports:
  - loopxDINO with Multi-step Cascade Loss across loops
  - Baseline-A (Linear), Baseline-B (2-layer Decoder), Baseline-C (6-layer Decoder)
  - Micro-batch accumulation on single 16GB RTX 5060 Ti GPU
  - Real-time mIoU and Boundary IoU evaluation
"""

import argparse
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from loopxdino.act import MultiStepCascadeLoss
from loopxdino.baselines import (
    BaselineA_Linear,
    BaselineB_TransformerDecoder,
    BaselineC_TransformerDecoder,
)
from loopxdino.data import get_voc_dataloaders
from loopxdino.loss import CombinedCEDiceLoss
from loopxdino.metrics import SegmentationEvaluator
from loopxdino.model import LoopxDINO


def build_model(
    model_type: str,
    num_classes: int = 21,
    num_loops: int = 4,
    force_mock: bool = True,
    device: str = "cuda",
) -> tuple[nn.Module, int]:
    """Builds and returns requested model with trainable parameter count."""
    if model_type == "loopxdino":
        model = LoopxDINO(
            backbone="facebook/dinov2-base",
            num_loops=num_loops,
            task_head="mask",
            num_classes=num_classes,
            force_mock_backbone=force_mock,
            use_checkpoint=True,
            use_mask_guidance=True,
            use_multiscale=True,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif model_type == "baseline_a":
        model = BaselineA_Linear(
            num_classes=num_classes,
            force_mock_backbone=force_mock,
        )
        trainable = model.decoder_params
    elif model_type == "baseline_b":
        model = BaselineB_TransformerDecoder(
            num_classes=num_classes,
            force_mock_backbone=force_mock,
        )
        trainable = model.decoder_params
    elif model_type == "baseline_c":
        model = BaselineC_TransformerDecoder(
            num_classes=num_classes,
            force_mock_backbone=force_mock,
        )
        trainable = model.decoder_params
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device, dtype=torch.bfloat16)
    return model, trainable


def train_one_epoch(
    model: nn.Module,
    model_type: str,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    cascade_loss_fn: MultiStepCascadeLoss,
    accum_steps: int,
    device: str,
    epoch: int,
    max_norm: float = 1.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    num_batches = len(dataloader)

    for batch_idx, (images, queries, targets) in enumerate(dataloader):
        images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _B = images.shape[0]

        # Use internal learned query bank if present, otherwise external queries
        if (
            hasattr(model, "learned_query_bank")
            and model.learned_query_bank is not None
        ):
            queries = model.learned_query_bank.expand(_B, -1, -1)
        else:
            queries = queries.to(device, dtype=torch.bfloat16, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if model_type == "loopxdino":
                out = model(images, queries, return_trajectory=True)
                step_masks = out["step_masks"]
                loss, _ = cascade_loss_fn(step_masks, targets)
            else:
                # Baseline forward
                mask_logits = model(images, queries)
                loss = criterion(mask_logits, targets)

            loss_scaled = loss / accum_steps

        loss_scaled.backward()
        total_loss += loss.item()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (batch_idx + 1) % max(1, (num_batches // 4)) == 0 or (
            batch_idx + 1
        ) == num_batches:
            vram_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else 0
            )
            print(
                f"  Epoch [{epoch}] | Batch [{batch_idx + 1:2d}/{num_batches:2d}] "
                f"| Loss: {loss.item():.4f} | Peak VRAM: {vram_mb:6.1f} MB",
                flush=True,
            )

    elapsed = time.time() - start_time
    avg_loss = total_loss / num_batches
    return avg_loss, elapsed


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    model_type: str,
    dataloader: DataLoader,
    device: str,
    num_classes: int = 21,
    num_loops: int | None = None,
) -> dict[str, float]:
    model.eval()
    evaluator = SegmentationEvaluator(num_classes=num_classes, ignore_index=255)

    for images, queries, targets in dataloader:
        images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _B = images.shape[0]

        if (
            hasattr(model, "learned_query_bank")
            and model.learned_query_bank is not None
        ):
            queries = model.learned_query_bank.expand(_B, -1, -1)
        else:
            queries = queries.to(device, dtype=torch.bfloat16, non_blocking=True)

        if model_type == "loopxdino":
            out = model(images, queries, num_loops=num_loops)
            if isinstance(out, tuple):
                mask_logits = out[0]
            elif isinstance(out, dict):
                mask_logits = (
                    out["output"][0]
                    if isinstance(out["output"], tuple)
                    else out["output"]
                )
            else:
                mask_logits = out
        else:
            mask_logits = model(images, queries)

        evaluator.update(mask_logits, targets)

    return evaluator.summary()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pascal VOC 2012 Dense Segmentation Trainer"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="loopxdino",
        choices=["loopxdino", "baseline_a", "baseline_b", "baseline_c"],
    )
    parser.add_argument(
        "--epochs", type=int, default=12, help="Number of training epochs"
    )
    parser.add_argument(
        "--dice_weight",
        type=float,
        default=1.0,
        help="Weight for MultiClassDiceLoss in hybrid loss",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Micro-batch size per iteration"
    )
    parser.add_argument(
        "--accum_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--num_loops", type=int, default=4, help="Reasoning loops for loopxDINO"
    )
    parser.add_argument(
        "--img_size", type=int, default=512, help="Image spatial resolution"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/pascal_voc", help="VOC dataset directory"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download official VOC 2012"
    )
    parser.add_argument(
        "--force_synthetic",
        action="store_true",
        default=False,
        help="Use synthetic VOC dataset for testing",
    )
    parser.add_argument(
        "--force_mock",
        action="store_true",
        default=False,
        help="Use mock ViT backbone instead of pretrained DINO",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(" Pascal VOC 2012 Segmentation Trainer", flush=True)
    print(
        f" Model: {args.model_type} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        flush=True,
    )
    print(
        f" Micro-batch: {args.batch_size} | Accumulation: {args.accum_steps} | Effective Batch: {args.batch_size * args.accum_steps}",
        flush=True,
    )
    print(f" Resolution: {args.img_size}x{args.img_size} | Precision: torch.bfloat16", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Data
    train_loader, val_loader = get_voc_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        download=args.download,
        force_synthetic=args.force_synthetic,
        num_workers=args.num_workers,
    )
    print(
        f"Dataset ready: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.",
        flush=True,
    )

    # 2. Build Model
    model, trainable_params = build_model(
        model_type=args.model_type,
        num_classes=21,
        num_loops=args.num_loops,
        force_mock=args.force_mock,
        device=device,
    )
    print(
        f"Trainable Decoder/Head Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)",
        flush=True,
    )

    # 3. Setup Optimizer & Loss Functions
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    criterion = CombinedCEDiceLoss(
        num_classes=21,
        ignore_index=255,
        ce_weight=1.0,
        dice_weight=args.dice_weight,
    )
    cascade_loss = MultiStepCascadeLoss(
        base_criterion=criterion,
        weighting_strategy="linear_ramp",
    )

    # 4. Training Loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        loss, duration = train_one_epoch(
            model=model,
            model_type=args.model_type,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            cascade_loss_fn=cascade_loss,
            accum_steps=args.accum_steps,
            device=device,
            epoch=epoch,
        )
        scheduler.step()

        # Validation
        val_metrics = evaluate_segmentation(
            model=model,
            model_type=args.model_type,
            dataloader=val_loader,
            device=device,
            num_classes=21,
        )
        print(
            f"Epoch [{epoch}/{args.epochs}] in {duration:.2f}s | Train Loss: {loss:.4f} "
            f"| Val mIoU: {val_metrics['mIoU']}% | Boundary IoU: {val_metrics['Boundary_IoU']}%\n",
            flush=True,
        )

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            ckpt_path = os.path.join(args.save_dir, f"{args.model_type}_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                ckpt_path,
            )

    print(f"Training finished successfully! Best Val mIoU: {best_miou:.2f}%\n", flush=True)

    # 5. For loopxDINO, evaluate test-time compute depth scaling
    if args.model_type == "loopxdino":
        best_ckpt = os.path.join(args.save_dir, f"{args.model_type}_best.pt")
        if os.path.exists(best_ckpt):
            ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state"])
            print(f"Loaded best checkpoint from epoch {ckpt_data.get('epoch', '?')} with Val mIoU: {ckpt_data.get('val_metrics', {}).get('mIoU', 0.0):.2f}% for test-time scaling analysis.", flush=True)

        print("=" * 70, flush=True)
        print(" Evaluating Test-Time Compute Depth Scaling (T = 1, 2, 4, 8, 16)...", flush=True)
        print("=" * 70, flush=True)
        for t in [1, 2, 4, 8, 16]:
            m = evaluate_segmentation(
                model=model,
                model_type=args.model_type,
                dataloader=val_loader,
                device=device,
                num_classes=21,
                num_loops=t,
            )
            print(
                f"  Test-Time Depth T = {t:2d} | Val mIoU: {m['mIoU']:5.2f}% | Boundary IoU: {m['Boundary_IoU']:5.2f}%",
                flush=True,
            )
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
