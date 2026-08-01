"""Tests for the beat-phase (frame-target-aware) augmentation pieces in
musicality.augmentations: FrameTimeStretch, BeatPhaseAugmenter,
AugmentedBeatDataset, build_beat_phase_augmenter.
"""

from unittest.mock import patch

import torch
import pytest
from omegaconf import OmegaConf

from musicality.augmentations import (
    AddNoise,
    AugmentedBeatDataset,
    BeatPhaseAugmenter,
    FrameTimeStretch,
    RandomGain,
    build_beat_phase_augmenter,
)

SR = 22050
N_SAMPLES = SR * 4  # 4 s
N_FRAMES = 200


def _wav_target():
    wav = torch.randn(1, N_SAMPLES)
    target = torch.zeros(4, N_FRAMES)
    target[0, ::10] = 1.0  # beat every 10 frames
    target[3] = 1.0  # mask on
    return wav, target


# ---------------------------------------------------------------------------
# FrameTimeStretch
# ---------------------------------------------------------------------------


class TestFrameTimeStretch:
    def test_wav_and_target_scale_by_same_ratio(self):
        wav, target = _wav_target()
        stretch = FrameTimeStretch(min_rate=1.2, max_rate=1.2)  # fixed rate

        with patch("musicality.augmentations.random.uniform", return_value=1.2):
            new_wav, new_target = stretch(wav, target, SR)

        wav_ratio = new_wav.shape[-1] / wav.shape[-1]
        target_ratio = new_target.shape[-1] / target.shape[-1]
        assert wav_ratio == pytest.approx(target_ratio, rel=0.02)

    def test_rate_greater_than_one_shrinks_length(self):
        wav, target = _wav_target()
        stretch = FrameTimeStretch()

        with patch("musicality.augmentations.random.uniform", return_value=1.5):
            new_wav, new_target = stretch(wav, target, SR)

        assert new_wav.shape[-1] < wav.shape[-1]
        assert new_target.shape[-1] < target.shape[-1]

    def test_rate_less_than_one_grows_length(self):
        wav, target = _wav_target()
        stretch = FrameTimeStretch()

        with patch("musicality.augmentations.random.uniform", return_value=0.7):
            new_wav, new_target = stretch(wav, target, SR)

        assert new_wav.shape[-1] > wav.shape[-1]
        assert new_target.shape[-1] > target.shape[-1]

    def test_target_channel_count_preserved(self):
        wav, target = _wav_target()
        stretch = FrameTimeStretch()

        with patch("musicality.augmentations.random.uniform", return_value=1.1):
            _, new_target = stretch(wav, target, SR)

        assert new_target.shape[0] == 4


# ---------------------------------------------------------------------------
# BeatPhaseAugmenter
# ---------------------------------------------------------------------------


class TestBeatPhaseAugmenter:
    def test_no_augmentations_returns_shapes_unchanged(self):
        wav, target = _wav_target()
        augmenter = BeatPhaseAugmenter()
        new_wav, new_target = augmenter(wav, target, SR, N_SAMPLES, N_FRAMES)

        assert torch.equal(new_wav, wav)
        assert torch.equal(new_target, target)

    def test_time_stretch_output_cropped_or_padded_to_fixed_lengths(self):
        wav, target = _wav_target()
        augmenter = BeatPhaseAugmenter(
            time_stretch=FrameTimeStretch(min_rate=0.7, max_rate=1.5)
        )

        for _ in range(10):  # several random rates, both shrink and grow cases
            new_wav, new_target = augmenter(wav, target, SR, N_SAMPLES, N_FRAMES)
            assert new_wav.shape == (1, N_SAMPLES)
            assert new_target.shape == (4, N_FRAMES)

    def test_gain_changes_wav_not_target(self):
        wav, target = _wav_target()
        augmenter = BeatPhaseAugmenter(gain=RandomGain(min_db=6.0, max_db=6.0))
        new_wav, new_target = augmenter(wav, target, SR, N_SAMPLES, N_FRAMES)

        assert not torch.equal(new_wav, wav)
        assert torch.equal(new_target, target)

    def test_noise_changes_wav_not_target(self):
        wav, target = _wav_target()
        augmenter = BeatPhaseAugmenter(noise=AddNoise(std=0.1))
        new_wav, new_target = augmenter(wav, target, SR, N_SAMPLES, N_FRAMES)

        assert not torch.equal(new_wav, wav)
        assert torch.equal(new_target, target)


# ---------------------------------------------------------------------------
# AugmentedBeatDataset
# ---------------------------------------------------------------------------


class TestAugmentedBeatDataset:
    def test_len_matches_subset(self):
        subset = [_wav_target() for _ in range(3)]
        ds = AugmentedBeatDataset(subset, BeatPhaseAugmenter(), SR, N_SAMPLES, N_FRAMES)
        assert len(ds) == 3

    def test_getitem_applies_augmenter(self):
        subset = [_wav_target()]
        augmenter = BeatPhaseAugmenter(gain=RandomGain(min_db=6.0, max_db=6.0))
        ds = AugmentedBeatDataset(subset, augmenter, SR, N_SAMPLES, N_FRAMES)

        wav, target = ds[0]
        orig_wav, orig_target = subset[0]

        assert not torch.equal(wav, orig_wav)
        assert torch.equal(target, orig_target)


# ---------------------------------------------------------------------------
# build_beat_phase_augmenter
# ---------------------------------------------------------------------------


class TestBuildBeatPhaseAugmenter:
    def test_returns_none_when_disabled(self):
        cfg = OmegaConf.create({"enabled": False})
        assert build_beat_phase_augmenter(cfg) is None

    def test_returns_none_when_all_suboptions_disabled(self):
        cfg = OmegaConf.create(
            {
                "enabled": True,
                "time_stretch": {"enabled": False},
                "gain": {"enabled": False},
                "noise": {"enabled": False},
            }
        )
        assert build_beat_phase_augmenter(cfg) is None

    def test_wires_frame_time_stretch(self):
        cfg = OmegaConf.create(
            {
                "enabled": True,
                "time_stretch": {"enabled": True, "min_rate": 0.8, "max_rate": 1.2},
                "gain": {"enabled": False},
                "noise": {"enabled": False},
            }
        )
        augmenter = build_beat_phase_augmenter(cfg)
        assert isinstance(augmenter, BeatPhaseAugmenter)
        assert isinstance(augmenter.time_stretch, FrameTimeStretch)
        assert augmenter.gain is None
        assert augmenter.noise is None

    def test_wires_gain_and_noise(self):
        cfg = OmegaConf.create(
            {
                "enabled": True,
                "time_stretch": {"enabled": False},
                "gain": {"enabled": True, "min_db": -3.0, "max_db": 3.0},
                "noise": {"enabled": True, "std": 0.01},
            }
        )
        augmenter = build_beat_phase_augmenter(cfg)
        assert augmenter.time_stretch is None
        assert isinstance(augmenter.gain, RandomGain)
        assert isinstance(augmenter.noise, AddNoise)
