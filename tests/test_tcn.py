"""Tests for musicality.models.tcn.TCNTempoNet — both pooled and frame-level modes."""

import torch
import pytest

from musicality.models.tcn import TCNTempoNet

N_SAMPLES = 4096  # short but > n_fft (2048) so STFT doesn't error


# ---------------------------------------------------------------------------
# Pooled (scalar / classification) mode — default, unchanged behavior
# ---------------------------------------------------------------------------


class TestPooledMode:
    def test_scalar_output_shape(self):
        model = TCNTempoNet(n_mels=16, channels=8, n_layers=3, n_outputs=1)
        out = model(torch.randn(4, 1, N_SAMPLES))
        assert out.shape == (4,)

    def test_classification_output_shape(self):
        model = TCNTempoNet(n_mels=16, channels=8, n_layers=3, n_outputs=64)
        out = model(torch.randn(4, 1, N_SAMPLES))
        assert out.shape == (4, 64)

    def test_single_sample(self):
        model = TCNTempoNet(n_mels=16, channels=8, n_layers=3, n_outputs=1)
        out = model(torch.randn(1, 1, N_SAMPLES))
        assert out.shape == (1,)

    def test_output_is_finite(self):
        model = TCNTempoNet(n_mels=16, channels=8, n_layers=3, n_outputs=1)
        out = model(torch.randn(4, 1, N_SAMPLES))
        assert torch.isfinite(out).all()

    def test_no_frame_head_allocated(self):
        """Pooled mode must not allocate the per-frame conv head."""
        model = TCNTempoNet(n_mels=16, channels=8, n_layers=3, n_outputs=1)
        assert not hasattr(model, "frame_head")
        assert hasattr(model, "head")


# ---------------------------------------------------------------------------
# Frame-level (beat-phase) mode
# ---------------------------------------------------------------------------


class TestFrameLevelMode:
    def test_output_shape_multi_channel(self):
        """3-channel frame-level output (e.g. beat/one/last) keeps the time axis."""
        model = TCNTempoNet(
            n_mels=16, channels=8, n_layers=3, n_outputs=3, frame_level=True
        )
        out = model(torch.randn(4, 1, N_SAMPLES))
        assert out.ndim == 3
        assert out.shape[0] == 4
        assert out.shape[1] == 3

    def test_output_shape_single_channel_squeezes(self):
        """n_outputs=1 in frame-level mode squeezes the channel dim, like pooled mode."""
        model = TCNTempoNet(
            n_mels=16, channels=8, n_layers=3, n_outputs=1, frame_level=True
        )
        out = model(torch.randn(4, 1, N_SAMPLES))
        assert out.shape == (4, out.shape[-1])

    def test_time_axis_matches_mel_frames(self):
        """Frame-level output length matches the mel transform's own frame count."""
        model = TCNTempoNet(
            n_mels=16, channels=8, n_layers=3, n_outputs=3, frame_level=True
        )
        wav = torch.randn(2, 1, N_SAMPLES)
        expected_t = model.mel(wav).squeeze(1).shape[-1]
        out = model(wav)
        assert out.shape[-1] == expected_t

    def test_output_is_finite(self):
        model = TCNTempoNet(
            n_mels=16, channels=8, n_layers=3, n_outputs=3, frame_level=True
        )
        out = model(torch.randn(4, 1, N_SAMPLES))
        assert torch.isfinite(out).all()

    def test_no_pooled_head_allocated(self):
        """Frame-level mode must not allocate the pooled FC head."""
        model = TCNTempoNet(
            n_mels=16, channels=8, n_layers=3, n_outputs=3, frame_level=True
        )
        assert not hasattr(model, "head")
        assert hasattr(model, "frame_head")

    def test_logits_not_prebounded(self):
        """Frame-level output is raw logits (no sigmoid applied) — values can exceed [0, 1]."""
        torch.manual_seed(0)
        model = TCNTempoNet(
            n_mels=16, channels=8, n_layers=3, n_outputs=3, frame_level=True
        )
        # Push the head's bias out of [0, 1] range to make the assertion meaningful.
        with torch.no_grad():
            model.frame_head.bias.fill_(5.0)
        out = model(torch.randn(2, 1, N_SAMPLES))
        assert (out > 1.0).any()

    def test_gradients_flow(self):
        model = TCNTempoNet(
            n_mels=16, channels=8, n_layers=3, n_outputs=3, frame_level=True
        )
        out = model(torch.randn(2, 1, N_SAMPLES))
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
