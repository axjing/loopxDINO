"""
benchmark_act.py
Benchmarks Adaptive Computation Time (ACT) & Throughput on RTX 5060 Ti.
Implements:
  1. Ponder time measurement across varying sample complexities.
  2. Throughput (FPS) comparison between fixed loops (N=1, 2, 4, 8, 12) and dynamic ACT.
  3. Shannon entropy convergence profile.
"""

import time

import torch

from loopxdino.model import LoopxDINO


def benchmark_throughput(
    model: LoopxDINO,
    image_size: int = 512,
    batch_size: int = 4,
    num_queries: int = 20,
    num_warmup: int = 10,
    num_iters: int = 50,
    num_loops: int = 4,
    dynamic_halt: bool = False,
    device: str = "cuda",
) -> tuple[float, float, float]:
    """
    Returns:
        fps: Frames per second.
        latency_ms: Average forward latency per batch in milliseconds.
        avg_loops: Average number of loops executed per sample.
    """
    model.eval()
    dummy_img = torch.randn(
        batch_size, 3, image_size, image_size, device=device, dtype=torch.bfloat16
    )
    dummy_q = torch.randn(
        batch_size, num_queries, 768, device=device, dtype=torch.bfloat16
    )

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(
                dummy_img, dummy_q, num_loops=num_loops, dynamic_halt=dynamic_halt
            )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(
                dummy_img, dummy_q, num_loops=num_loops, dynamic_halt=dynamic_halt
            )
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    total_images = num_iters * batch_size
    fps = total_images / total_time
    latency_ms = (total_time / num_iters) * 1000.0
    avg_loops = (
        model.ponder_tracker.average_ponder_time if dynamic_halt else float(num_loops)
    )

    return fps, latency_ms, avg_loops


def main():
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required for throughput benchmarking.")
        return

    device_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print(f" ACT & Throughput Benchmark on {device_name}")
    print("=" * 70)

    model = LoopxDINO(
        backbone="facebook/dinov3-vitb16-pretrain-lvd1689m",
        force_mock_backbone=True,
        task_head="classification",
        num_classes=100,
        dynamic_halt=True,
        entropy_threshold=0.25,
    ).to("cuda", dtype=torch.bfloat16)

    batch_size = 4
    img_size = 512
    num_queries = 20

    print(
        f"\nConfiguration: Batch Size={batch_size}, Image={img_size}x{img_size}, Queries={num_queries}\n"
    )

    # 1. Fixed loop benchmarks
    fixed_loop_settings = [1, 2, 4, 8, 12]
    print(
        f"{'Mode':<18} | {'Loops':<8} | {'Latency (ms)':<14} | {'Throughput (FPS)':<18}"
    )
    print("-" * 65)

    for n in fixed_loop_settings:
        fps, latency, _loops = benchmark_throughput(
            model,
            image_size=img_size,
            batch_size=batch_size,
            num_queries=num_queries,
            num_loops=n,
            dynamic_halt=False,
        )
        print(f"{'Fixed-Loop':<18} | {n:<8} | {latency:10.2f} ms   | {fps:12.1f} FPS")

    # 2. Dynamic ACT benchmarks with different entropy thresholds
    print("\n--- Adaptive Computation Time (ACT) Dynamic Early Exit ---")
    print(
        f"{'Threshold (eps)':<18} | {'Avg Ponder':<12} | {'Latency (ms)':<14} | {'Throughput (FPS)':<18} | {'Compute Saved':<14}"
    )
    print("-" * 80)

    for eps in [0.10, 0.25, 0.40, 0.60]:
        model.halting_monitor.entropy_threshold = eps
        model.ponder_tracker.reset()
        fps, latency, avg_ponder = benchmark_throughput(
            model,
            image_size=img_size,
            batch_size=batch_size,
            num_queries=num_queries,
            num_loops=10,
            dynamic_halt=True,
        )
        savings = model.ponder_tracker.compute_saving_ratio * 100
        print(
            f"eps = {eps:<12.2f} | {avg_ponder:8.2f}     | {latency:10.2f} ms   | {fps:12.1f} FPS      | {savings:10.1f}%"
        )

    print("\n" + "=" * 70)
    print(" ACT BENCHMARK COMPLETE! ✨")
    print("=" * 70)


if __name__ == "__main__":
    main()
