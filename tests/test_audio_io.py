"""Tests for musicality.loaders.audio_io — the seek-based crop reader shared
by the dataset loaders. Fixtures are real WAV files on disk, since the whole
point of this module is what it does with a file handle.
"""

import random

import numpy as np
import pytest
import soundfile as sf
import torch

SAMPLE_RATE = 22050
DURATION = 10.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)


def _write_wav(path, data, sr):
    sf.write(path, data, sr, subtype="PCM_16")
    return str(path)


def _noise(duration, sr, channels=1):
    return (np.random.randn(int(duration * sr), channels) * 0.1).astype(np.float32)


def _ramp(duration, sr):
    """A monotonically increasing signal, so a crop's content identifies its offset."""

    n = int(duration * sr)
    return np.linspace(-0.9, 0.9, n, dtype=np.float32).reshape(-1, 1)


class TestLoadCrop:
    def test_exact_length_at_native_rate(self, tmp_path):
        from musicality.loaders.audio_io import load_crop

        path = _write_wav(tmp_path / "a.wav", _noise(30.0, SAMPLE_RATE), SAMPLE_RATE)
        wav, _ = load_crop(path, SAMPLE_RATE, N_SAMPLES)

        assert wav.shape == (1, N_SAMPLES)
        assert wav.dtype == torch.float32

    def test_exact_length_after_resample(self, tmp_path):
        """A 44.1 kHz source still yields exactly n_samples at 22.05 kHz."""
        from musicality.loaders.audio_io import load_crop

        path = _write_wav(tmp_path / "a.wav", _noise(30.0, 44100), 44100)
        wav, _ = load_crop(path, SAMPLE_RATE, N_SAMPLES)

        assert wav.shape == (1, N_SAMPLES)

    def test_middle_crop_start(self, tmp_path):
        from musicality.loaders.audio_io import load_crop

        path = _write_wav(tmp_path / "a.wav", _noise(30.0, SAMPLE_RATE), SAMPLE_RATE)
        _, start_seconds = load_crop(path, SAMPLE_RATE, N_SAMPLES, crop="middle")

        assert start_seconds == pytest.approx((30.0 - DURATION) / 2, abs=1e-3)

    def test_start_crop_start(self, tmp_path):
        from musicality.loaders.audio_io import load_crop

        path = _write_wav(tmp_path / "a.wav", _noise(30.0, SAMPLE_RATE), SAMPLE_RATE)
        _, start_seconds = load_crop(path, SAMPLE_RATE, N_SAMPLES, crop="start")

        assert start_seconds == 0.0

    def test_random_crop_varies_and_stays_in_bounds(self, tmp_path):
        from musicality.loaders.audio_io import load_crop

        path = _write_wav(tmp_path / "a.wav", _noise(30.0, SAMPLE_RATE), SAMPLE_RATE)

        random.seed(0)
        starts = {
            load_crop(path, SAMPLE_RATE, N_SAMPLES, crop="random")[1] for _ in range(15)
        }

        assert len(starts) > 1
        assert all(0.0 <= s <= 30.0 - DURATION + 1e-6 for s in starts)

    def test_crop_content_matches_requested_window(self, tmp_path):
        """The returned samples are the file's samples at the reported offset."""
        from musicality.loaders.audio_io import load_crop

        data = _ramp(30.0, SAMPLE_RATE)
        path = _write_wav(tmp_path / "a.wav", data, SAMPLE_RATE)

        wav, start_seconds = load_crop(path, SAMPLE_RATE, N_SAMPLES, crop="middle")
        start = int(round(start_seconds * SAMPLE_RATE))
        expected = data[start : start + N_SAMPLES, 0]

        assert np.allclose(wav[0].numpy(), expected, atol=1e-4)

    def test_short_file_is_zero_padded(self, tmp_path):
        from musicality.loaders.audio_io import load_crop

        path = _write_wav(tmp_path / "a.wav", _noise(2.0, SAMPLE_RATE), SAMPLE_RATE)
        wav, start_seconds = load_crop(path, SAMPLE_RATE, N_SAMPLES, crop="random")

        n_real = int(2.0 * SAMPLE_RATE)

        assert wav.shape == (1, N_SAMPLES)
        assert start_seconds == 0.0
        assert torch.all(wav[0, n_real:] == 0.0)
        assert torch.any(wav[0, :n_real] != 0.0)

    def test_stereo_is_mixed_down_to_mono(self, tmp_path):
        from musicality.loaders.audio_io import load_crop

        left = (np.random.randn(int(15.0 * SAMPLE_RATE)) * 0.1).astype(np.float32)
        stereo = np.stack([left, -left], axis=1)
        path = _write_wav(tmp_path / "a.wav", stereo, SAMPLE_RATE)

        wav, _ = load_crop(path, SAMPLE_RATE, N_SAMPLES)

        assert wav.shape == (1, N_SAMPLES)
        # L and R cancel exactly, up to PCM_16 quantization.
        assert wav.abs().max().item() < 1e-3

    def test_unreadable_file_raises_runtime_error(self, tmp_path):
        from musicality.loaders.audio_io import load_crop

        path = tmp_path / "not_audio.wav"
        path.write_text("definitely not a wav")

        with pytest.raises(RuntimeError):
            load_crop(str(path), SAMPLE_RATE, N_SAMPLES)


class TestCropStartFrame:
    def test_track_shorter_than_crop_starts_at_zero(self):
        from musicality.loaders.audio_io import crop_start_frame

        for mode in ("start", "middle", "random"):
            assert crop_start_frame(100, 500, mode) == 0

    def test_middle_is_centered(self):
        from musicality.loaders.audio_io import crop_start_frame

        assert crop_start_frame(1000, 400, "middle") == 300

    def test_start_is_zero(self):
        from musicality.loaders.audio_io import crop_start_frame

        assert crop_start_frame(1000, 400, "start") == 0


class TestResample:
    def test_same_rate_is_a_passthrough(self):
        from musicality.loaders.audio_io import resample

        wav = torch.randn(1, 1000)

        assert resample(wav, SAMPLE_RATE, SAMPLE_RATE) is wav

    def test_kernel_is_cached_per_rate_pair(self):
        from musicality.loaders import audio_io

        audio_io._RESAMPLERS.clear()
        wav = torch.randn(1, 4410)

        audio_io.resample(wav, 44100, SAMPLE_RATE)
        first = audio_io._RESAMPLERS[(44100, SAMPLE_RATE)]
        audio_io.resample(wav, 44100, SAMPLE_RATE)

        assert audio_io._RESAMPLERS[(44100, SAMPLE_RATE)] is first

    def test_output_length_scales_with_rate_ratio(self):
        from musicality.loaders.audio_io import resample

        wav = torch.randn(1, 44100)
        out = resample(wav, 44100, SAMPLE_RATE)

        assert out.shape[1] == pytest.approx(22050, abs=2)
