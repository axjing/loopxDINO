import torch

from core.backbone import DINOBackbone


def test_backbone_frozen():
    backbone = DINOBackbone(force_mock=True)
    # Check that all parameters have requires_grad = False
    for p in backbone.parameters():
        assert not p.requires_grad, "Backbone parameters must have requires_grad=False"


def test_backbone_stays_in_eval():
    backbone = DINOBackbone(force_mock=True)
    backbone.train(True)
    assert not backbone.training, "Backbone must never be in training mode"
    assert not backbone.model.training, "Underlying model must remain in eval mode"


def test_backbone_forward_shape():
    backbone = DINOBackbone(embed_dim=768, patch_size=16, force_mock=True)
    x = torch.randn(2, 3, 256, 256)
    tokens, grid = backbone(x, return_patch_grid=True)
    assert grid == (16, 16)
    assert tokens.shape == (2, 256, 768)


def test_backbone_zero_grad():
    backbone = DINOBackbone(force_mock=True)
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    out = backbone(x)
    # Since forward is wrapped with @torch.no_grad(), out does not require grad
    assert not out.requires_grad, "Backbone output must never require gradients"
    for p in backbone.parameters():
        assert p.grad is None, "Backbone parameters must have no gradients"
