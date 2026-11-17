import pytest
import torch

from core.model import LoopxDINO


def test_loopxdino_forward_backward_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = LoopxDINO(
        backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_loops=4,
        force_mock_backbone=True,
    ).to("cuda", dtype=torch.bfloat16)

    model.train()
    img = torch.randn(2, 3, 256, 256, device="cuda", dtype=torch.bfloat16)
    queries = torch.randn(
        2, 8, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    out = model(img, queries)
    assert out.shape == (2, 8, 768)

    loss = out.sum()
    loss.backward()

    # Backbone must have zero gradients
    for p in model.backbone.parameters():
        assert p.grad is None

    # Decoder parameters must receive gradients
    decoder_grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert len(decoder_grads) > 0


def test_loopxdino_dense_mask_segmentation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = LoopxDINO(
        backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_loops=3,
        force_mock_backbone=True,
        task_head="mask",
        num_classes=21,
    ).to(device, dtype=dtype)
    model.eval()

    img = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
    queries = torch.randn(1, 5, 768, device=device, dtype=dtype)

    with torch.no_grad():
        mask_logits, class_logits = model(img, queries)

    assert mask_logits.shape == (1, 5, 256, 256)
    assert class_logits.shape == (1, 5, 21)


def test_loopxdino_dynamic_act_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = LoopxDINO(
        backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_loops=8,
        force_mock_backbone=True,
        task_head="classification",
        num_classes=10,
        dynamic_halt=True,
        entropy_threshold=0.5,
    ).to(device, dtype=dtype)
    model.eval()

    img = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
    queries = torch.randn(1, 4, 768, device=device, dtype=dtype)

    with torch.no_grad():
        out = model(img, queries, return_trajectory=True)

    assert "ponder_steps" in out
    assert out["ponder_steps"] >= 1
    assert out["ponder_steps"] <= 8


def test_loopxdino_learned_query_bank():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = LoopxDINO(
        backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_loops=2,
        force_mock_backbone=True,
        task_head="mask",
        num_classes=21,
    ).to(device, dtype=dtype)

    img = torch.randn(2, 3, 256, 256, device=device, dtype=dtype)
    # Forward without explicit queries -> should use internal learned query bank
    mask_logits, class_logits = model(img)
    assert mask_logits.shape == (2, 21, 256, 256)
    assert class_logits.shape == (2, 21, 21)

    loss = mask_logits.sum()
    loss.backward()
    assert model.learned_query_bank.grad is not None
