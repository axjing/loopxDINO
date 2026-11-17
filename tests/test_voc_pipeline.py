import torch

from core.data import SyntheticVOCDataset, get_voc_dataloaders
from core.metrics import SegmentationEvaluator, extract_boundary_band


def test_synthetic_voc_dataset():
    ds = SyntheticVOCDataset(num_samples=8, img_size=256, num_queries=21, embed_dim=768)
    assert len(ds) == 8
    img, queries, mask = ds[0]
    assert img.shape == (3, 256, 256)
    assert queries.shape == (21, 768)
    assert mask.shape == (256, 256)
    assert mask.dtype == torch.long


def test_get_voc_dataloaders():
    train_loader, val_loader = get_voc_dataloaders(
        batch_size=2,
        img_size=128,
        force_synthetic=True,
    )
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    batch_img, batch_q, batch_m = next(iter(train_loader))
    assert batch_img.shape == (2, 3, 128, 128)
    assert batch_q.shape == (2, 21, 768)
    assert batch_m.shape == (2, 128, 128)


def test_boundary_band_extraction():
    mask = torch.zeros(2, 64, 64, dtype=torch.long)
    mask[:, 20:40, 20:40] = 1
    boundary = extract_boundary_band(mask, dilation_radius=2)
    assert boundary.shape == (2, 64, 64)
    assert boundary.sum() > 0
    assert (boundary == 1).sum() < 64 * 64


def test_segmentation_evaluator():
    evaluator = SegmentationEvaluator(num_classes=21, ignore_index=255)
    # Perfect prediction test
    target = torch.randint(0, 5, (2, 32, 32))
    target[0, :5, :5] = 255  # ignore region
    pred_logits = torch.zeros(2, 21, 32, 32)
    for b in range(2):
        for h in range(32):
            for w in range(32):
                cls_idx = target[b, h, w].item()
                if cls_idx < 21:
                    pred_logits[b, cls_idx, h, w] = 10.0

    evaluator.update(pred_logits, target)
    summary = evaluator.summary()
    assert summary["mIoU"] > 95.0
    assert summary["Boundary_IoU"] > 90.0
