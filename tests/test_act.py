import torch

from core.act import (
    DifferentiableACTController,
    MultiStepCascadeLoss,
    PonderLoss,
    PonderTracker,
    ShannonEntropyHalting,
    compute_shannon_entropy,
)


def test_shannon_entropy_normalization():
    # Uniform distribution has maximum entropy (1.0 normalized)
    logits_uniform = torch.zeros(2, 5, 100)
    entropy_max = compute_shannon_entropy(logits_uniform, normalize=True)
    assert torch.allclose(entropy_max, torch.ones_like(entropy_max), atol=1e-4)

    # Sharp distribution has low entropy
    logits_sharp = torch.zeros(2, 5, 100)
    logits_sharp[:, :, 0] = 50.0
    entropy_low = compute_shannon_entropy(logits_sharp, normalize=True)
    assert (entropy_low < 0.05).all()


def test_shannon_entropy_halting():
    monitor = ShannonEntropyHalting(entropy_threshold=0.20, min_loops=1, max_loops=6)

    # Step 0 with high entropy -> should not halt
    logits_high = torch.randn(2, 10, 50)
    halt, _, _ = monitor.should_halt(logits_high, current_step=0)
    assert not halt

    # Step 2 with very sharp logits -> should halt early
    logits_sharp = torch.zeros(2, 10, 50)
    logits_sharp[:, :, 0] = 30.0
    halt, _, _ = monitor.should_halt(logits_sharp, current_step=2)
    assert halt

    # At max_loops - 1 -> should always halt
    halt, _, _ = monitor.should_halt(logits_high, current_step=5)
    assert halt


def test_multi_step_cascade_loss():
    criterion = MultiStepCascadeLoss()
    pred_step1 = torch.randn(4, 10)
    pred_step2 = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))

    loss, loss_dict = criterion([pred_step1, pred_step2], targets)
    assert loss.dim() == 0
    assert "step_1_loss" in loss_dict
    assert "step_2_loss" in loss_dict
    assert "total_cascade_loss" in loss_dict


def test_ponder_tracker():
    tracker = PonderTracker(max_loops=10)
    tracker.record(steps_taken=4, mean_entropy=0.15, batch_size=2)
    tracker.record(steps_taken=6, mean_entropy=0.18, batch_size=2)

    assert tracker.total_samples == 4
    assert tracker.total_steps == 20
    assert tracker.average_ponder_time == 5.0
    assert tracker.compute_saving_ratio == 0.50


def test_differentiable_act_controller():
    controller = DifferentiableACTController(d_model=64, max_loops=5)
    states = [torch.randn(2, 4, 64) for _ in range(4)]
    weights, ponder_cost, accumulated_state = controller(states)

    assert weights.shape == (2, 4)
    # Weights should sum to 1 per sample
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-4)
    assert ponder_cost.shape == (2,)
    assert (ponder_cost >= 1.0).all()
    assert accumulated_state.shape == (2, 4, 64)


def test_ponder_loss():
    base_loss = torch.nn.CrossEntropyLoss()
    criterion = PonderLoss(task_criterion=base_loss, tau=0.01)
    preds = torch.randn(4, 10, requires_grad=True)
    targets = torch.randint(0, 10, (4,))
    ponder_cost = torch.tensor([2.5, 3.0, 1.5, 4.0])

    loss, loss_dict = criterion(preds, targets, ponder_cost)
    assert loss.dim() == 0
    assert "task_loss" in loss_dict
    assert "ponder_penalty" in loss_dict
    assert "total_loss" in loss_dict
    loss.backward()
    assert preds.grad is not None
