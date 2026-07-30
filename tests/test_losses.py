"""Tests for musicality.losses.beat_phase_loss."""

import torch
import pytest

from musicality.losses import beat_phase_loss

B, T = 4, 50


def _logits(value: float = 0.0) -> torch.Tensor:
    return torch.full((B, 3, T), value, requires_grad=True)


def _target(beat=0.0, one=0.0, four=0.0, mask=1.0) -> torch.Tensor:
    t = torch.zeros(B, 4, T)
    t[:, 0] = beat
    t[:, 1] = one
    t[:, 2] = four
    t[:, 3] = mask
    return t


class TestBeatPhaseLoss:
    def test_output_is_scalar(self):
        loss = beat_phase_loss(_logits(), _target())
        assert loss.shape == ()

    def test_loss_is_positive(self):
        loss = beat_phase_loss(_logits(), _target(beat=1.0, one=1.0, four=1.0))
        assert loss.item() > 0.0

    def test_perfect_prediction_lower_than_wrong(self):
        target = _target(beat=1.0, one=1.0, four=1.0)

        correct_logits = _logits(10.0)  # sigmoid(10) ~ 1.0, matches target
        wrong_logits = _logits(-10.0)  # sigmoid(-10) ~ 0.0, opposite of target

        correct_loss = beat_phase_loss(correct_logits, target)
        wrong_loss = beat_phase_loss(wrong_logits, target)

        assert correct_loss.item() < wrong_loss.item()

    def test_gradients_flow(self):
        logits = _logits()
        loss = beat_phase_loss(logits, _target(beat=1.0, one=1.0, four=1.0))
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_mask_zero_ignores_one_four_errors(self):
        """When mask=0, one/four predictions must not affect the loss at all."""
        logits = _logits(0.0)

        target_masked_wrong = _target(beat=1.0, one=1.0, four=1.0, mask=0.0)
        target_masked_matching_beat_only = _target(
            beat=1.0, one=0.0, four=0.0, mask=0.0
        )

        loss_a = beat_phase_loss(logits, target_masked_wrong)
        loss_b = beat_phase_loss(logits, target_masked_matching_beat_only)

        assert loss_a.item() == pytest.approx(loss_b.item())

    def test_all_masked_out_batch_has_no_nan(self):
        """A batch where every track lacks position annotations must not divide by zero."""
        loss = beat_phase_loss(_logits(), _target(beat=1.0, mask=0.0))
        assert torch.isfinite(loss).all()

    def test_per_head_pos_weight(self):
        """A shape-(3,) pos_weight applies independently per head."""
        target = _target(beat=1.0, one=1.0, four=1.0)
        logits = _logits(-2.0)  # confidently wrong on all heads

        low_weight = beat_phase_loss(
            logits, target, pos_weight=torch.tensor([1.0, 1.0, 1.0])
        )
        high_beat_weight = beat_phase_loss(
            logits, target, pos_weight=torch.tensor([20.0, 1.0, 1.0])
        )

        assert high_beat_weight.item() > low_weight.item()
