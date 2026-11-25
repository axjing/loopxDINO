import torch

from loopxdino.baselines import (
    BaselineA_Linear,
    BaselineB_TransformerDecoder,
    BaselineC_TransformerDecoder,
)


def test_baseline_a_linear():
    model = BaselineA_Linear(num_classes=21, force_mock_backbone=True)
    assert model.decoder_params < 5_000_000
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 21, 128, 128)

    # Backbone is frozen
    for p in model.backbone.parameters():
        assert not p.requires_grad


def test_baseline_b_transformer():
    model = BaselineB_TransformerDecoder(num_classes=21, force_mock_backbone=True)
    assert model.decoder.num_layers == 2
    x = torch.randn(2, 3, 128, 128)
    q = torch.randn(2, 21, 768)
    out = model(x, q)
    assert out.shape == (2, 21, 128, 128)


def test_baseline_c_transformer():
    model = BaselineC_TransformerDecoder(num_classes=21, force_mock_backbone=True)
    assert model.decoder.num_layers == 6
    x = torch.randn(1, 3, 128, 128)
    q = torch.randn(1, 21, 768)
    out = model(x, q)
    assert out.shape == (1, 21, 128, 128)
