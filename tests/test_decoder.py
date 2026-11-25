import torch

from loopxdino.decoder import LoopedCrossAttentionDecoder


def test_decoder_parameter_budget():
    decoder = LoopedCrossAttentionDecoder(
        d_model=768, nhead=12, dim_feedforward=2048, max_loops=16
    )
    total_params = decoder.total_params
    assert total_params < 10_000_000, (
        f"Decoder params ({total_params}) must be strictly < 10M"
    )


def test_step_embedding_differentiation():
    decoder = LoopedCrossAttentionDecoder(d_model=768, max_loops=8)
    decoder.eval()
    queries = torch.randn(2, 5, 768)
    visual = torch.randn(2, 64, 768)

    with torch.no_grad():
        _, traj = decoder(queries, visual, num_loops=3, return_trajectory=True)

    # traj[0] is Q_0, traj[1] is Q_1, traj[2] is Q_2, traj[3] is Q_3
    assert len(traj) == 4
    # Query must change across steps
    assert not torch.allclose(traj[0], traj[1])
    assert not torch.allclose(traj[1], traj[2])
    assert not torch.allclose(traj[2], traj[3])


def test_kv_caching():
    decoder = LoopedCrossAttentionDecoder(d_model=768)
    visual = torch.randn(2, 64, 768)
    k1, v1 = decoder.precompute_kv(visual)
    k2, v2 = decoder.precompute_kv(visual)
    assert torch.allclose(k1, k2)
    assert torch.allclose(v1, v2)
    assert k1.shape == (2, 64, 768)
    assert v1.shape == (2, 64, 768)


def test_attention_weights_extraction():
    decoder = LoopedCrossAttentionDecoder(d_model=768)
    queries = torch.randn(1, 4, 768)
    visual = torch.randn(1, 64, 768)
    _out, attn_maps = decoder(queries, visual, num_loops=3, return_attn_weights=True)
    assert len(attn_maps) == 3
    for attn in attn_maps:
        assert attn.shape == (1, 4, 64)
        # Probabilities should sum to ~1 over visual keys
        assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 4), atol=1e-4)


def test_adaln_step_modulation_and_norm_stability():
    # adaLN mode
    decoder_adaln = LoopedCrossAttentionDecoder(
        d_model=768, max_loops=16, step_mode="adaln"
    )
    assert decoder_adaln.total_params < 10_000_000
    assert hasattr(decoder_adaln, "step_mod")

    queries = torch.randn(2, 4, 768, requires_grad=True)
    visual = torch.randn(2, 32, 768)

    out, traj = decoder_adaln(queries, visual, num_loops=8, return_trajectory=True)
    assert len(traj) == 9
    assert out.shape == (2, 4, 768)

    # Backward gradient check
    loss = out.sum()
    loss.backward()
    assert decoder_adaln.step_mod.weight.grad is not None
    assert decoder_adaln.residual_alpha.grad is not None

    # Check norm stability across 8 loops (should remain well-bounded)
    norms = [q.norm(dim=-1).mean().item() for q in traj]
    # Norms should remain stable rather than exploding monotonically
    assert all(n < 50.0 for n in norms)
