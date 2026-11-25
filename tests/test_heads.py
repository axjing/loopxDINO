import torch

from loopxdino.heads import ClassificationHead, DenseMaskHead


def test_dense_mask_head():
    for mod_mode in ["global", "per_query"]:
        head = DenseMaskHead(
            d_model=768,
            num_classes=21,
            feat_dim=128,
            patch_size=16,
            modulation_mode=mod_mode,
        )
        queries = torch.randn(2, 5, 768)
        visual = torch.randn(2, 256, 768)
        grid_size = (16, 16)

        mask_logits, class_logits = head(
            queries, visual, grid_size=grid_size, out_size=(256, 256)
        )
        assert mask_logits.shape == (2, 5, 256, 256)
        assert class_logits.shape == (2, 5, 21)


def test_classification_head():
    head_mean = ClassificationHead(d_model=768, num_classes=10, pool_mode="mean")
    queries = torch.randn(2, 5, 768)
    out_mean = head_mean(queries)
    assert out_mean.shape == (2, 10)

    head_token = ClassificationHead(d_model=768, num_classes=10, pool_mode="none")
    out_token = head_token(queries)
    assert out_token.shape == (2, 5, 10)
