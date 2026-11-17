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

from core.act import MultiStepCascadeLoss
from core.baselines import (
    BaselineA_Linear,
    BaselineB_TransformerDecoder,
    BaselineC_TransformerDecoder,
)
from core.data import get_voc_dataloaders
from core.metrics import SegmentationEvaluator
from core.model import LoopxDINO


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
            backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
            num_loops=num_loops,
            task_head="mask",
            num_classes=num_classes,
            force_mock_backbone=force_mock,
            use_checkpoint=True,
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
        _B, _C, H, W = images.shape

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
                # System 1: visual tokens
                visual_memory, grid_size = model.extract_visual_features(
                    images, return_grid=True
                )
                # System 2: trajectory of queries across loops
                decoder_out = model.decoder(
                    queries=queries,
                    visual_memory=visual_memory,
                    num_loops=model.num_loops,
                    return_trajectory=True,
                )
                _, trajectory = decoder_out

                # Project each intermediate reasoning loop to dense segmentation masks
                step_masks = []
                for q_step in trajectory[1:]:  # loops 1 to N
                    mask_logits, _ = model.task_head(
                        q_step, visual_memory, grid_size, out_size=(H, W)
                    )
                    step_masks.append(mask_logits)

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
                f"| Loss: {loss.item():.4f} | Peak VRAM: {vram_mb:6.1f} MB"
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
            out = model(images, queries)
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
        "--epochs", type=int, default=2, help="Number of training epochs"
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
        default=True,
        help="Use synthetic VOC dataset for testing",
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
    print("=" * 70)
    print(" Pascal VOC 2012 Segmentation Trainer")
    print(
        f" Model: {args.model_type} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(
        f" Micro-batch: {args.batch_size} | Accumulation: {args.accum_steps} | Effective Batch: {args.batch_size * args.accum_steps}"
    )
    print(f" Resolution: {args.img_size}x{args.img_size} | Precision: torch.bfloat16")
    print("=" * 70)

    # 1. Load Data
    train_loader, val_loader = get_voc_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        download=args.download,
        force_synthetic=args.force_synthetic,
    )
    print(
        f"Dataset ready: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples."
    )

    # 2. Build Model
    model, trainable_params = build_model(
        model_type=args.model_type,
        num_classes=21,
        num_loops=args.num_loops,
        force_mock=True,
        device=device,
    )
    print(
        f"Trainable Decoder/Head Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)"
    )

    # 3. Setup Optimizer & Loss Functions
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    ce_criterion = nn.CrossEntropyLoss(ignore_index=255)
    cascade_loss = MultiStepCascadeLoss(
        base_criterion=ce_criterion,
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
            criterion=ce_criterion,
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
            f"| Val mIoU: {val_metrics['mIoU']}% | Boundary IoU: {val_metrics['Boundary_IoU']}%\n"
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

    print(f"Training finished successfully! Best Val mIoU: {best_miou:.2f}%\n")


if __name__ == "__main__":
    main()
