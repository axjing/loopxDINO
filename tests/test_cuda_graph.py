import pytest
import torch

from loopxdino.cuda_graph import LoopedCUDAGraphRunner
from loopxdino.decoder import LoopedCrossAttentionDecoder


def test_cuda_graph_runner():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CUDA Graph testing")

    decoder = LoopedCrossAttentionDecoder(d_model=768, max_loops=4).to(
        "cuda", dtype=torch.bfloat16
    )

    runner = LoopedCUDAGraphRunner(
        decoder=decoder,
        batch_size=1,
        num_queries=10,
        num_visual_tokens=64,
        d_model=768,
        num_loops=3,
        device="cuda",
        dtype=torch.bfloat16,
    )

    assert runner.is_captured

    q = torch.randn(1, 10, 768, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 64, 768, device="cuda", dtype=torch.bfloat16)

    # Replay graph
    out_graph = runner(q, v)
    assert out_graph.shape == (1, 10, 768)

    # Verify eager fallback on different batch size
    q_diff = torch.randn(2, 10, 768, device="cuda", dtype=torch.bfloat16)
    v_diff = torch.randn(2, 64, 768, device="cuda", dtype=torch.bfloat16)
    out_fallback = runner(q_diff, v_diff)
    assert out_fallback.shape == (2, 10, 768)
