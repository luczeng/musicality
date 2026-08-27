"""Tests for musicality.models.tcn.TCNTempoNet — both pooled and frame-level modes."""

import math

import torch
import torch.nn as nn
import pytest

from musicality.models.tcn import TCNTempoNet, PositionalEncoding, SelfAttentionBlock

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
        # frame_head is Sequential(Dropout, Conv1d) — index 1 is the Conv1d.
        with torch.no_grad():
            model.frame_head[1].bias.fill_(5.0)
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


# ---------------------------------------------------------------------------
# Self-attention head building blocks (beat-phase one/last long-range context
# — see docs/beat_phase_context_ideas.md)
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    def test_shape_preserved(self):
        pe = PositionalEncoding(channels=8)
        x = torch.randn(4, 20, 8)
        out = pe(x)
        assert out.shape == x.shape

    def test_output_is_finite(self):
        pe = PositionalEncoding(channels=8)
        out = pe(torch.randn(4, 20, 8))
        assert torch.isfinite(out).all()

    def test_encoding_is_additive_and_input_independent(self):
        """The positional component added must depend only on position and
        channel, never on the input's values."""
        pe = PositionalEncoding(channels=8)
        x1 = torch.randn(2, 10, 8)
        x2 = torch.randn(2, 10, 8)
        added1 = pe(x1) - x1
        added2 = pe(x2) - x2
        assert torch.allclose(added1, added2, atol=1e-6)

    def test_varies_across_time_positions(self):
        """Different positions must get different encodings, or every frame
        looks identical to the attention block regardless of order."""
        pe = PositionalEncoding(channels=8)
        out = pe(torch.zeros(1, 10, 8))
        assert not torch.allclose(out[0, 0], out[0, 5])

    def test_handles_varying_sequence_lengths(self):
        pe = PositionalEncoding(channels=8)
        short = pe(torch.randn(1, 5, 8))
        long = pe(torch.randn(1, 50, 8))
        assert short.shape == (1, 5, 8)
        assert long.shape == (1, 50, 8)

    def test_rejects_odd_channels(self):
        """Standard sin/cos pairing needs an even channel count."""
        with pytest.raises(Exception):
            PositionalEncoding(channels=7)

    def test_matches_reference_formula(self):
        """Pins down the actual sin/cos values against a hand-computed reference,
        rather than just structural properties — catches axis-order and
        off-by-one bugs that shape/variation checks alone miss."""
        channels = 4
        seq_len = 6
        pe = PositionalEncoding(channels=channels)
        out = pe(torch.zeros(1, seq_len, channels))

        expected = torch.zeros(seq_len, channels)
        for pos in range(seq_len):
            for k in range(channels):
                angle = pos / (10000 ** (2 * (k // 2) / channels))
                expected[pos, k] = math.sin(angle) if k % 2 == 0 else math.cos(angle)

        assert torch.allclose(out[0], expected, atol=1e-4)


class TestSelfAttentionBlock:
    def test_shape_preserved(self):
        block = SelfAttentionBlock(channels=8, n_heads=2)
        x = torch.randn(4, 20, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_output_is_finite(self):
        block = SelfAttentionBlock(channels=8, n_heads=2)
        out = block(torch.randn(4, 20, 8))
        assert torch.isfinite(out).all()

    def test_output_differs_from_input(self):
        block = SelfAttentionBlock(channels=8, n_heads=2)
        x = torch.randn(4, 20, 8)
        out = block(x)
        assert not torch.allclose(out, x)

    def test_gradients_flow(self):
        block = SelfAttentionBlock(channels=8, n_heads=2)
        x = torch.randn(4, 20, 8, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        for p in block.parameters():
            assert p.grad is not None

    def test_mixes_information_across_time(self):
        """A real self-attention layer lets every frame's output depend on every
        other frame's input — perturbing one frame should move others' outputs.
        This is what actually buys 'longer range than the TCN's fixed window'
        (see docs/beat_phase_context_ideas.md)."""
        torch.manual_seed(0)
        block = SelfAttentionBlock(channels=8, n_heads=2)
        block.eval()  # disable dropout so the comparison isn't noisy

        x = torch.randn(1, 10, 8)
        out1 = block(x)

        x2 = x.clone()
        x2[0, 0] += 5.0  # perturb only frame 0
        out2 = block(x2)

        # frame 5's output should move too, since it attended to frame 0
        assert not torch.allclose(out1[0, 5], out2[0, 5], atol=1e-4)

    def test_uses_independent_layer_norms(self):
        """The attention sublayer and the feedforward sublayer each need their
        own LayerNorm — reusing one instance for both forces them to share the
        same learnable scale/shift, which isn't how the design is supposed to
        work (see the two separate 'residual + LayerNorm' steps in the spec)."""
        block = SelfAttentionBlock(channels=8, n_heads=2)
        norms = [m for m in block.modules() if isinstance(m, nn.LayerNorm)]
        assert len(norms) == 2
        assert norms[0] is not norms[1]


class TestSelfAttentionIntegration:
    def test_default_behavior_unchanged(self):
        """use_self_attention defaults to False — must allocate the original
        single frame_head, not the scoped beat/phase heads."""
        model = TCNTempoNet(
            n_mels=16, channels=8, n_layers=3, n_outputs=3, frame_level=True
        )
        assert hasattr(model, "frame_head")
        assert not hasattr(model, "beat_head")
        assert not hasattr(model, "phase_head")

    def test_scoped_heads_allocated(self):
        model = TCNTempoNet(
            n_mels=16,
            channels=8,
            n_layers=3,
            n_outputs=3,
            frame_level=True,
            use_self_attention=True,
        )
        assert hasattr(model, "beat_head")
        assert hasattr(model, "phase_head")
        assert not hasattr(model, "frame_head")

    def test_output_shape_unchanged(self):
        model = TCNTempoNet(
            n_mels=16,
            channels=8,
            n_layers=3,
            n_outputs=3,
            frame_level=True,
            use_self_attention=True,
            n_attn_layers=1,
            n_attn_heads=2,
        )
        out = model(torch.randn(4, 1, N_SAMPLES))
        assert out.shape[0] == 4
        assert out.shape[1] == 3

    def test_gradients_flow(self):
        model = TCNTempoNet(
            n_mels=16,
            channels=8,
            n_layers=3,
            n_outputs=3,
            frame_level=True,
            use_self_attention=True,
        )
        out = model(torch.randn(2, 1, N_SAMPLES))
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)

    def test_beat_channel_is_independent_of_attention_params(self):
        """Scoping requirement: `beat` must read straight off the trunk, so its
        loss gradient must not reach the self-attention block's parameters —
        only `one`/`last` should route through it (see
        docs/beat_phase_context_ideas.md's placement notes)."""
        model = TCNTempoNet(
            n_mels=16,
            channels=8,
            n_layers=3,
            n_outputs=3,
            frame_level=True,
            use_self_attention=True,
        )
        out = model(torch.randn(2, 1, N_SAMPLES))
        out[:, 0, :].sum().backward()

        attn_params = list(model.phase_head.parameters())
        assert len(attn_params) > 0
        for p in attn_params:
            assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))

    def test_phase_channels_depend_on_attention_params(self):
        """Sanity check the opposite direction: one/last must actually be wired
        through the attention block, not disconnected from it."""
        model = TCNTempoNet(
            n_mels=16,
            channels=8,
            n_layers=3,
            n_outputs=3,
            frame_level=True,
            use_self_attention=True,
        )
        out = model(torch.randn(2, 1, N_SAMPLES))
        out[:, 1:, :].sum().backward()

        attn_params = list(model.phase_head.parameters())
        assert len(attn_params) > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn_params)
