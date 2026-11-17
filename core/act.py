"""
Adaptive Computation Time (ACT) & Shannon Entropy Halting
Implements:
  1. Shannon entropy-based dynamic halting for test-time adaptive compute.
  2. Multi-step cascade loss for intermediate loop supervision (cascade distillation).
  3. Ponder tracker to measure computational savings and dynamic depth allocation.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def compute_shannon_entropy(
    logits: torch.Tensor,
    dim: int = -1,
    normalize: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Computes Shannon entropy:
      H(p) = - sum(p * log(p + eps))
    If normalize is True, H is normalized by log(num_classes) into [0, 1].
    Low entropy indicates sharp, highly confident predictions.
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = torch.log(probs + eps)
    entropy = -(probs * log_probs).sum(dim=dim)

    if normalize:
        num_classes = logits.shape[dim]
        if num_classes > 1:
            max_entropy = math.log(num_classes)
            entropy = entropy / max_entropy
    return entropy


class ShannonEntropyHalting(nn.Module):
    """
    Monitors prediction entropy at each loop iteration.
    Signals early stopping when mean token entropy falls below halting threshold epsilon.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.20,
        min_loops: int = 1,
        max_loops: int = 10,
        reduction: str = "mean",
    ):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.min_loops = min_loops
        self.max_loops = max_loops
        self.reduction = reduction

    def should_halt(
        self,
        logits: torch.Tensor,
        current_step: int,
    ) -> tuple[bool, torch.Tensor, float]:
        """
        Determines whether iterative reasoning should halt.

        Args:
            logits: Output logits at current step, shape (B, L_q, C) or (B, C).
            current_step: 0-indexed loop step (i.e. step = 0 means 1st loop completed).

        Returns:
            halt: Boolean flag indicating if halting condition is met.
            sample_entropies: Normalized entropy tensor per batch item.
            mean_entropy: Scalar mean entropy across batch.
        """
        actual_loop = current_step + 1

        # Must run at least min_loops
        if actual_loop < self.min_loops:
            return False, torch.zeros(logits.shape[0], device=logits.device), 1.0

        # Maximum loop limit reached
        if actual_loop >= self.max_loops:
            return True, torch.zeros(logits.shape[0], device=logits.device), 0.0

        # Compute normalized entropy
        token_entropy = compute_shannon_entropy(logits, dim=-1, normalize=True)
        # Reduce across tokens to get per-sample entropy
        if token_entropy.dim() > 1:
            sample_entropy = token_entropy.mean(dim=-1)
        else:
            sample_entropy = token_entropy

        mean_entropy = sample_entropy.mean().item()
        halt = mean_entropy <= self.entropy_threshold
        return halt, sample_entropy, mean_entropy


class MultiStepCascadeLoss(nn.Module):
    """
    Multi-Step Cascade Loss for System-2 training:
    Supervises predictions from every loop step t in {1, ..., N}.
    Ensures that iterative reasoning progressively refines the representation.

      L_total = sum_{t=1}^N w_t * L(pred_t, target)
    """

    def __init__(
        self,
        base_criterion: nn.Module | None = None,
        weighting_strategy: str = "linear_ramp",  # 'uniform', 'linear_ramp', 'exponential'
    ):
        super().__init__()
        self.base_criterion = (
            base_criterion if base_criterion is not None else nn.CrossEntropyLoss()
        )
        self.weighting_strategy = weighting_strategy

    def get_step_weights(self, num_steps: int, device: torch.device) -> torch.Tensor:
        if self.weighting_strategy == "uniform":
            weights = torch.ones(num_steps, device=device) / num_steps
        elif self.weighting_strategy == "linear_ramp":
            # Steps later in the loop carry higher weight: 1, 2, ..., N
            steps = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)
            weights = steps / steps.sum()
        elif self.weighting_strategy == "exponential":
            steps = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)
            weights = torch.exp(steps * 0.5)
            weights = weights / weights.sum()
        else:
            weights = torch.ones(num_steps, device=device) / num_steps
        return weights

    def forward(
        self,
        step_predictions: list[torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            step_predictions: List of predictions [P_1, P_2, ..., P_N].
            target: Ground truth tensor.

        Returns:
            total_loss: Aggregated weighted scalar loss.
            loss_dict: Dictionary with individual step losses for logging.
        """
        N = len(step_predictions)
        if N == 0:
            raise ValueError("step_predictions list must not be empty.")

        device = step_predictions[0].device
        weights = self.get_step_weights(N, device)

        total_loss = torch.tensor(0.0, device=device, dtype=step_predictions[0].dtype)
        loss_dict = {}

        for t, pred in enumerate(step_predictions):
            step_loss = self.base_criterion(pred, target)
            total_loss = total_loss + weights[t] * step_loss
            loss_dict[f"step_{t + 1}_loss"] = step_loss.item()

        loss_dict["total_cascade_loss"] = total_loss.item()
        return total_loss, loss_dict


class PonderTracker:
    """
    Tracks ponder time (loop steps) and computation statistics during ACT inference.
    """

    def __init__(self, max_loops: int = 10):
        self.max_loops = max_loops
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.total_steps = 0
        self.step_counts = [0] * (self.max_loops + 1)
        self.entropies = []

    def record(self, steps_taken: int, mean_entropy: float, batch_size: int = 1):
        self.total_samples += batch_size
        self.total_steps += steps_taken * batch_size
        if steps_taken <= self.max_loops:
            self.step_counts[steps_taken] += batch_size
        self.entropies.append(mean_entropy)

    @property
    def average_ponder_time(self) -> float:
        return self.total_steps / max(1, self.total_samples)

    @property
    def compute_saving_ratio(self) -> float:
        """Percentage of compute saved compared to running max_loops always."""
        theoretical_max = self.total_samples * self.max_loops
        if theoretical_max == 0:
            return 0.0
        return 1.0 - (self.total_steps / theoretical_max)

    def summary(self) -> dict[str, float | int | list[int]]:
        return {
            "total_samples": self.total_samples,
            "average_ponder_time": round(self.average_ponder_time, 2),
            "compute_saving_pct": round(self.compute_saving_ratio * 100, 2),
            "step_distribution": self.step_counts,
        }


class DifferentiableACTController(nn.Module):
    """
    Graves-style Differentiable Adaptive Computation Time (ACT):
    Predicts scalar halting probability h_t in [0, 1] for each step t.
    Computes step probabilities p_t and differentiable ponder cost.
    """

    def __init__(self, d_model: int, max_loops: int = 10, epsilon: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.max_loops = max_loops
        self.epsilon = epsilon
        # Halting unit: predicts scalar probability from query states
        self.halting_unit = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        step_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            step_states: List of intermediate states [Q_1, Q_2, ..., Q_T], each of shape (B, L_q, d_model).

        Returns:
            step_weights: Soft weights p_t assigned to each step, shape (B, T).
            ponder_cost: Ponder cost per sample, shape (B,).
            accumulated_state: Ponder-weighted query states, shape (B, L_q, d_model).
        """
        B = step_states[0].shape[0]
        device = step_states[0].device
        T = len(step_states)

        halting_probs = []
        for state in step_states:
            # Pool across tokens to get sample-level halting probability
            h = self.halting_unit(state.mean(dim=1)).squeeze(-1)  # (B,)
            halting_probs.append(h)

        halting_probs = torch.stack(halting_probs, dim=1)  # (B, T)

        # Compute cumulative probabilities and Graves weights
        cum_prob = torch.zeros(B, device=device)
        weights = torch.zeros(B, T, device=device)
        halting_step = torch.full((B,), T, dtype=torch.long, device=device)
        still_running = torch.ones(B, dtype=torch.bool, device=device)

        for t in range(T):
            h_t = halting_probs[:, t]
            new_cum = cum_prob + h_t

            # For samples that halt at step t
            halt_now = (new_cum >= 1.0 - self.epsilon) & still_running
            # Remainder
            remainder = (1.0 - cum_prob).clamp(min=0.0)

            weights[:, t] = torch.where(
                halt_now,
                remainder,
                torch.where(still_running, h_t, torch.zeros_like(h_t)),
            )

            halting_step = torch.where(
                halt_now, torch.full_like(halting_step, t + 1), halting_step
            )
            still_running = still_running & (~halt_now)
            cum_prob = new_cum

        # If any sample never reached 1 - eps by step T, assign remainder to final step
        unhalted = cum_prob < 1.0 - self.epsilon
        if unhalted.any():
            weights[unhalted, -1] += (1.0 - cum_prob[unhalted]).clamp(min=0.0)

        # Ponder cost: sum of steps weighted by halting probabilities + remainder
        ponder_cost = halting_step.float() + (1.0 - cum_prob).clamp(min=0.0)

        # Weighted combination of states
        accumulated_state = torch.zeros_like(step_states[0])
        for t, s in enumerate(step_states):
            accumulated_state = accumulated_state + weights[:, t].view(B, 1, 1) * s

        return weights, ponder_cost, accumulated_state


class PonderLoss(nn.Module):
    """
    Combined Task and Ponder Loss for Graves ACT:
      L = L_task + tau * Ponder_Cost
    """

    def __init__(self, task_criterion: nn.Module, tau: float = 0.01):
        super().__init__()
        self.task_criterion = task_criterion
        self.tau = tau

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ponder_cost: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        task_loss = self.task_criterion(predictions, targets)
        ponder_penalty = self.tau * ponder_cost.mean()
        total_loss = task_loss + ponder_penalty
        return total_loss, {
            "task_loss": task_loss.item(),
            "ponder_penalty": ponder_penalty.item(),
            "total_loss": total_loss.item(),
            "mean_ponder_steps": ponder_cost.mean().item(),
        }
