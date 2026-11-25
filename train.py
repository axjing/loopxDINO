"""
train.py - Minimalist High-Performance Training Loop for loopxDINO
Optimized for single 16GB RTX 5060 Ti GPU.
Features:
  - Micro-batch accumulation to simulate large effective batch sizes.
  - Native torch.bfloat16 mixed precision.
  - Multi-step cascade loss across all reasoning loops.
  - Strictly frozen perception backbone (eval mode, requires_grad=False).
  - VRAM tracking and gradient clipping.
"""

import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from loopxdino.act import MultiStepCascadeLoss
from loopxdino.model import LoopxDINO


class SyntheticDenseDataset(Dataset):
    """
    Synthetic dataset for dense vision reasoning verification and benchmarking.
    Generates synthetic images, queries, and class targets.
    """

    def __init__(
        self,
        num_samples: int = 128,
        img_size: int = 256,
        num_queries: int = 10,
        num_classes: int = 21,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_queries = num_queries
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        image = torch.randn(3, self.img_size, self.img_size)
        queries = torch.randn(self.num_queries, 768)
        target = torch.randint(0, self.num_classes, ())
        return image, queries, target


def train_epoch(
    model: LoopxDINO,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiStepCascadeLoss,
    accum_steps: int,
    device: str,
    epoch: int,
    max_norm: float = 1.0,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()
    num_batches = len(dataloader)

    for batch_idx, (images, queries, targets) in enumerate(dataloader):
        images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
        queries = queries.to(device, dtype=torch.bfloat16, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Run model with return_trajectory=True for multi-step cascade loss
            outputs = model(
                images,
                queries,
                return_trajectory=True,
            )
            trajectory = outputs["trajectory"]  # List of [Q_0, Q_1, ..., Q_N]

            # In classification mode: pass each step query through classifier
            step_logits = []
            for q_step in trajectory[1:]:  # Supervise steps 1 to N
                logits = model.task_head(
                    q_step
                )  # (B, num_classes) or (B, L_q, num_classes)
                if logits.dim() == 2:
                    step_logits.append(logits)
                else:
                    step_logits.append(logits.view(-1, logits.shape[-1]))

            flat_targets = (
                targets.view(-1)
                if step_logits[0].dim() == 2 and targets.dim() > 1
                else targets
            )
            if step_logits[0].shape[0] != flat_targets.shape[0]:
                flat_targets = flat_targets.repeat(
                    step_logits[0].shape[0] // flat_targets.shape[0]
                )

            loss, _loss_dict = criterion(step_logits, flat_targets)
            loss_scaled = loss / accum_steps

        loss_scaled.backward()
        total_loss += loss.item()

        # Step optimizer upon reaching accumulation boundary
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
            # Gradient clipping for numerical stability
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


def parse_args():
    parser = argparse.ArgumentParser(description="loopxDINO Minimal Training Loop")
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Physical micro-batch size per iteration",
    )
    parser.add_argument(
        "--accum_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for AdamW"
    )
    parser.add_argument(
        "--num_loops",
        type=int,
        default=4,
        help="Reasoning loops (System 2 compute depth)",
    )
    parser.add_argument(
        "--img_size", type=int, default=256, help="Input image dimension (H=W)"
    )
    parser.add_argument(
        "--num_queries", type=int, default=10, help="Number of query tokens"
    )
    parser.add_argument(
        "--num_classes", type=int, default=21, help="Number of semantic classes"
    )
    parser.add_argument(
        "--samples", type=int, default=32, help="Number of dataset samples for demo"
    )
    parser.add_argument(
        "--force_mock",
        action="store_true",
        default=True,
        help="Force mock backbone for fast offline training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(" loopxDINO System-2 Training Session")
    print(
        f" Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(
        f" Micro-batch: {args.batch_size} | Accumulation Steps: {args.accum_steps} "
        f"| Effective Batch: {args.batch_size * args.accum_steps}"
    )
    print(
        f" Reasoning Loops: {args.num_loops} | Checkpoint: Enabled | Precision: bfloat16"
    )
    print("=" * 65)

    # Instantiate model
    model = LoopxDINO(
        backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_loops=args.num_loops,
        force_mock_backbone=args.force_mock,
        task_head="classification",
        num_classes=args.num_classes,
        use_checkpoint=True,
    ).to(device, dtype=torch.bfloat16)

    # Strict parameter filtering: only decoder and head parameters get optimized
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    assert len(backbone_params) == 0, (
        "Backbone parameters must have requires_grad=False!"
    )

    trainable_num = sum(p.numel() for p in trainable_params)
    print(
        f"Trainable parameters: {trainable_num:,} ({trainable_num / 1e6:.2f}M) - Backbone frozen (0 grad)"
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    criterion = MultiStepCascadeLoss(
        base_criterion=nn.CrossEntropyLoss(),
        weighting_strategy="linear_ramp",
    )

    dataset = SyntheticDenseDataset(
        num_samples=args.samples,
        img_size=args.img_size,
        num_queries=args.num_queries,
        num_classes=args.num_classes,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    print(f"\nStarting training for {args.epochs} epoch(s)...")
    for epoch in range(1, args.epochs + 1):
        loss, duration = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            accum_steps=args.accum_steps,
            device=device,
            epoch=epoch,
        )
        print(
            f"Epoch {epoch} finished in {duration:.2f}s | Average Cascade Loss: {loss:.4f}\n"
        )

    print("Training finished successfully! All invariants preserved. 🎉")


if __name__ == "__main__":
    main()
