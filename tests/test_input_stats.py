"""Tests for per-band input normalisation (musicality/input_stats.py).

The defect being fixed is a train/inference mismatch: the old whole-input
normalisation made the same audio arrive at the network differently depending on
how much audio surrounded it. The central test here is
:meth:`TestLengthInvariance.test_crop_matches_same_window_inside_a_long_input`,
which asserts the mismatch is gone; everything else guards the plumbing.
"""

import pytest
import torch
import torch.nn as nn
import torchaudio.transforms as T

from musicality.input_stats import (
    MIN_STD,
    compute_band_stats,
    fit_input_stats,
    waveform_batches,
)
from musicality.models.tcn import TCNTempoNet


N_MELS = 16
SAMPLE_RATE = 22050


def _mel(n_mels: int = N_MELS) -> nn.Module:
    return nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=n_mels, n_fft=512, hop_length=256
        ),
        T.AmplitudeToDB(),
    )


def _model(input_norm: str = "fixed", n_mels: int = N_MELS) -> TCNTempoNet:
    return TCNTempoNet(
        n_mels=n_mels,
        sample_rate=SAMPLE_RATE,
        hop_length=256,
        channels=8,
        n_layers=3,
        n_outputs=1,
        frame_level=True,
        input_norm=input_norm,
    )


class TestComputeBandStats:
    def test_returns_one_statistic_per_band(self):
        mean, std = compute_band_stats(_mel(), [torch.randn(2, 1, 8000)])

        assert mean.shape == (N_MELS,)
        assert std.shape == (N_MELS,)

    def test_matches_a_direct_computation(self):
        mel = _mel()
        wav = torch.randn(3, 1, 8000)

        mean, std = compute_band_stats(mel, [wav])

        reference = mel(wav).reshape(-1, N_MELS, mel(wav).shape[-1])
        expected_mean = reference.mean(dim=(0, 2))
        expected_std = reference.std(dim=(0, 2), correction=0)

        assert torch.allclose(mean, expected_mean, atol=1e-3)
        assert torch.allclose(std, expected_std, atol=1e-3)

    def test_batching_does_not_change_the_result(self):
        """Accumulated as sums, not as a mean of per-batch means — so an uneven
        split into batches must give the same answer as one big batch."""

        mel = _mel()
        wav = torch.randn(6, 1, 8000)

        one_batch = compute_band_stats(mel, [wav])
        split = compute_band_stats(mel, [wav[:1], wav[1:5], wav[5:]])

        assert torch.allclose(one_batch[0], split[0], atol=1e-4)
        assert torch.allclose(one_batch[1], split[1], atol=1e-4)

    def test_accepts_unchanneled_waveforms(self):
        mean, _ = compute_band_stats(_mel(), [torch.randn(2, 8000)])

        assert mean.shape == (N_MELS,)

    def test_stops_after_max_batches(self):
        seen = []

        def counting():
            for i in range(10):
                seen.append(i)
                yield torch.randn(1, 1, 8000)

        compute_band_stats(_mel(), counting(), max_batches=3)

        assert len(seen) == 3

    def test_silent_band_gets_a_floored_std(self):
        """A constant band has zero variance; dividing by it would turn its
        noise floor into signal."""

        mel = nn.Identity()
        constant = torch.full((1, 4, 100), -80.0)

        _, std = compute_band_stats(mel, [constant])

        assert torch.all(std >= MIN_STD)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="no batches"):
            compute_band_stats(_mel(), [])


class TestWaveformBatches:
    def test_unwraps_tuples(self):
        wav = torch.randn(2, 1, 100)
        target = torch.randn(2, 3, 10)

        assert list(waveform_batches([(wav, target)]))[0] is wav

    def test_passes_bare_tensors_through(self):
        wav = torch.randn(2, 1, 100)

        assert list(waveform_batches([wav]))[0] is wav


class TestSetInputStats:
    def test_marks_fitted(self):
        model = _model()
        assert not model.input_stats_fitted

        model.set_input_stats(torch.zeros(N_MELS), torch.ones(N_MELS))

        assert model.input_stats_fitted

    def test_rejects_wrong_shape(self):
        model = _model()

        with pytest.raises(ValueError, match="shape"):
            model.set_input_stats(torch.zeros(N_MELS + 1), torch.ones(N_MELS + 1))

    def test_rejects_non_positive_std(self):
        model = _model()
        std = torch.ones(N_MELS)
        std[0] = 0.0

        with pytest.raises(ValueError, match="std"):
            model.set_input_stats(torch.zeros(N_MELS), std)

    def test_rejects_global_models(self):
        model = _model(input_norm="global")

        with pytest.raises(RuntimeError, match="input_norm='fixed'"):
            model.set_input_stats(torch.zeros(N_MELS), torch.ones(N_MELS))

    def test_global_model_is_never_fitted(self):
        assert not _model(input_norm="global").input_stats_fitted


class TestFitInputStats:
    def test_installs_statistics(self):
        model = _model()

        result = fit_input_stats(model, [(torch.randn(2, 1, 8000), None)])

        assert result is not None
        assert model.input_stats_fitted
        assert torch.allclose(model.norm_mean, result[0])

    def test_is_a_noop_for_global_models(self):
        model = _model(input_norm="global")

        assert fit_input_stats(model, [(torch.randn(2, 1, 8000), None)]) is None

    def test_does_not_overwrite_already_fitted_statistics(self):
        """Resuming a run, or continuing from a checkpoint, must keep the
        statistics the weights were trained against."""

        model = _model()
        model.set_input_stats(torch.full((N_MELS,), 7.0), torch.full((N_MELS,), 2.0))

        assert fit_input_stats(model, [(torch.randn(2, 1, 8000), None)]) is None
        assert torch.allclose(model.norm_mean, torch.full((N_MELS,), 7.0))


class TestConstruction:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="input_norm"):
            _model(input_norm="per_band")

    def test_global_model_registers_no_buffers(self):
        names = dict(_model(input_norm="global").named_buffers())

        assert "norm_mean" not in names

    def test_fixed_model_registers_buffers_before_fitting(self):
        """They must exist unconditionally, or a checkpoint's state_dict and a
        freshly constructed model disagree on their keys."""

        names = dict(_model().named_buffers())

        assert {"norm_mean", "norm_std", "norm_fitted"} <= set(names)

    def test_unfitted_statistics_are_the_identity(self):
        model = _model()

        assert torch.allclose(model.norm_mean, torch.zeros(N_MELS))
        assert torch.allclose(model.norm_std, torch.ones(N_MELS))


class TestLengthInvariance:
    """The point of the whole change."""

    def test_crop_matches_same_window_inside_a_long_input(self):
        model = _model().eval()
        model.set_input_stats(torch.full((N_MELS,), 3.0), torch.full((N_MELS,), 9.0))

        long_wav = torch.randn(1, 1, 256 * 400)
        start, length = 256 * 50, 256 * 100
        crop = long_wav[..., start : start + length]

        with torch.no_grad():
            from_crop = model(crop)
            from_long = model(long_wav)[..., 50 : 50 + 100]

        # Convolution padding differs at the crop's edges, so compare the
        # interior, well inside the trunk's receptive field.
        assert torch.allclose(from_crop[..., 30:70], from_long[..., 30:70], atol=1e-4)

    def test_global_normalisation_does_not_have_that_property(self):
        """Guards the characterisation of the bug: if this ever starts passing,
        the premise of the fix has changed."""

        model = _model(input_norm="global").eval()

        long_wav = torch.cat(
            [torch.randn(1, 1, 256 * 300) * 0.01, torch.randn(1, 1, 256 * 100)], dim=-1
        )
        crop = long_wav[..., 256 * 300 :]

        with torch.no_grad():
            from_crop = model(crop)
            from_long = model(long_wav)[..., 300:]

        assert not torch.allclose(
            from_crop[..., 30:70], from_long[..., 30:70], atol=1e-3
        )


class TestCheckpointRoundTrip:
    def test_statistics_survive_a_state_dict_round_trip(self):
        trained = _model()
        trained.set_input_stats(
            torch.arange(N_MELS).float(), torch.full((N_MELS,), 4.0)
        )

        restored = _model()
        restored.load_state_dict(trained.state_dict())

        assert restored.input_stats_fitted
        assert torch.allclose(restored.norm_mean, trained.norm_mean)
        assert torch.allclose(restored.norm_std, trained.norm_std)

    def test_restored_model_matches_the_original(self):
        trained = _model().eval()
        trained.set_input_stats(torch.full((N_MELS,), 2.0), torch.full((N_MELS,), 5.0))

        restored = _model().eval()
        restored.load_state_dict(trained.state_dict())

        wav = torch.randn(1, 1, 8000)
        with torch.no_grad():
            assert torch.allclose(trained(wav), restored(wav), atol=1e-6)

    def test_global_checkpoint_does_not_load_into_a_fixed_model(self):
        """Different normalisation is a different model; the key mismatch is the
        guard that stops one silently loading as the other."""

        with pytest.raises(RuntimeError):
            _model().load_state_dict(_model(input_norm="global").state_dict())
