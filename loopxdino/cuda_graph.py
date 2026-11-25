"""
CUDA Graph Static Execution Engine for loopxDINO
Captures looped reasoning decoder passes into static CUDA Graphs on NVIDIA GPUs (RTX 5060 Ti).
Eliminates Python CPU dispatch and kernel launch overhead, enabling sub-2.5ms inference.
"""

import torch

from loopxdino.decoder import LoopedCrossAttentionDecoder


class LoopedCUDAGraphRunner:
    """
    Manages CUDA Graph capture and replay for LoopedCrossAttentionDecoder.

    Args:
        decoder: Instance of LoopedCrossAttentionDecoder.
        batch_size: Fixed batch size for static capture.
        num_queries: Number of query tokens (L_q).
        num_visual_tokens: Number of visual patch tokens (S_v).
        d_model: Hidden feature dimension.
        num_loops: Number of reasoning iterations to capture.
        device: CUDA device (e.g. 'cuda' or 'cuda:0').
        dtype: Tensor precision (torch.bfloat16 or torch.float16).
    """

    def __init__(
        self,
        decoder: LoopedCrossAttentionDecoder,
        batch_size: int = 1,
        num_queries: int = 21,
        num_visual_tokens: int = 256,
        d_model: int = 768,
        num_loops: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.num_visual_tokens = num_visual_tokens
        self.d_model = d_model
        self.num_loops = num_loops
        self.device = torch.device(device)
        self.dtype = dtype

        self.is_captured = False
        self.graph: torch.cuda.CUDAGraph | None = None

        # Static I/O buffers
        self.static_queries: torch.Tensor | None = None
        self.static_visual: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None

        if torch.cuda.is_available() and self.device.type == "cuda":
            self._capture_graph()

    def _capture_graph(self) -> None:
        """Performs warmup and captures the CUDA Graph."""
        self.decoder.eval()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(device=self.device))

        with torch.cuda.stream(stream):
            # 1. Allocate static buffers
            self.static_queries = torch.randn(
                self.batch_size,
                self.num_queries,
                self.d_model,
                device=self.device,
                dtype=self.dtype,
            )
            self.static_visual = torch.randn(
                self.batch_size,
                self.num_visual_tokens,
                self.d_model,
                device=self.device,
                dtype=self.dtype,
            )

            # 2. Warmup passes to initialize CUDA state and memory pool
            for _ in range(3):
                _ = self.decoder(
                    self.static_queries,
                    self.static_visual,
                    num_loops=self.num_loops,
                    return_trajectory=False,
                )

            # 3. Capture graph
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.static_output = self.decoder(
                    self.static_queries,
                    self.static_visual,
                    num_loops=self.num_loops,
                    return_trajectory=False,
                )

        torch.cuda.current_stream(device=self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.is_captured = True

    @torch.no_grad()
    def __call__(
        self,
        queries: torch.Tensor,
        visual_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes inference via CUDA Graph replay or eager fallback.

        Args:
            queries: Input queries of shape (B, L_q, d_model).
            visual_memory: Visual memory of shape (B, S_v, d_model).

        Returns:
            Refined features tensor.
        """
        # Shape check for CUDA Graph match
        if (
            self.is_captured
            and self.graph is not None
            and queries.shape == self.static_queries.shape
            and visual_memory.shape == self.static_visual.shape
            and queries.dtype == self.dtype
        ):
            self.static_queries.copy_(queries)
            self.static_visual.copy_(visual_memory)
            self.graph.replay()
            return self.static_output.clone()

        # Fallback to eager execution
        return self.decoder(
            queries,
            visual_memory,
            num_loops=self.num_loops,
            return_trajectory=False,
        )
