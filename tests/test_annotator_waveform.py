"""Tests for the pure compute_envelope helper in waveform_widget."""

import numpy as np

from tools.annotator.waveform_widget import compute_envelope


class TestComputeEnvelope:
    def test_output_length(self):
        audio = np.random.randn(44100)
        min_env, max_env = compute_envelope(audio, n_points=512)
        assert len(min_env) == 512
        assert len(max_env) == 512

    def test_min_leq_max(self):
        audio = np.random.randn(22050)
        min_env, max_env = compute_envelope(audio)
        assert np.all(min_env <= max_env)

    def test_empty_audio_returns_zeros(self):
        min_env, max_env = compute_envelope(np.array([]), n_points=64)
        assert len(min_env) == 64
        assert len(max_env) == 64
        assert np.all(min_env == 0)
        assert np.all(max_env == 0)

    def test_constant_signal(self):
        audio = np.ones(22050) * 0.5
        min_env, max_env = compute_envelope(audio)
        np.testing.assert_allclose(min_env, 0.5)
        np.testing.assert_allclose(max_env, 0.5)

    def test_bounds_within_signal_range(self):
        audio = np.random.uniform(-0.8, 0.8, 22050)
        min_env, max_env = compute_envelope(audio)
        assert min_env.min() >= -0.8
        assert max_env.max() <= 0.8
