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
from musicality.loaders.beat_dataset import MAX_BEATS, PAD_TIME

SR = 22050
N_SAMPLES = SR * 4  # 4 s
N_FRAMES = 200


def _wav_target():
    wav = torch.randn(1, N_SAMPLES)
    target = torch.zeros(4, N_FRAMES)
    target[0, ::10] = 1.0  # beat every 10 frames
    target[3] = 1.0  # mask on
    return wav, target


def _beats():
    """The ``beats`` tensor matching :func:`_wav_target`'s beat every 10 frames."""

    beats = torch.zeros(2, MAX_BEATS)
    beats[0] = PAD_TIME
    beats[0, :20] = torch.arange(0, N_FRAMES, 10, dtype=torch.float32)
    beats[1, :20] = torch.arange(20) % 4 + 1
    return beats


def _peaks(channel):
    """Frame of the maximum of each contiguous nonzero run in *channel*."""

    peaks, start = [], None
    for i, value in enumerate([*channel.tolist(), 0.0]):
        if value > 0 and start is None:
            start = i
        elif value == 0 and start is not None:
            peaks.append(start + int(channel[start:i].argmax()))
            start = None
    return torch.tensor(peaks, dtype=torch.float32)


# ---------------------------------------------------------------------------
# FrameTimeStretch
# ---------------------------------------------------------------------------


class TestFrameTimeStretch:
    def test_wav_and_target_scale_by_same_ratio(self):
        wav, target = _wav_target()
        stretch = FrameTimeStretch(min_rate=1.2, max_rate=1.2)  # fixed rate

        with patch("musicality.augmentations.random.uniform", return_value=1.2):
            new_wav, new_target, _ = stretch(wav, target, SR)

        wav_ratio = new_wav.shape[-1] / wav.shape[-1]
        target_ratio = new_target.shape[-1] / target.shape[-1]
        assert wav_ratio == pytest.approx(target_ratio, rel=0.02)

    def test_rate_greater_than_one_shrinks_length(self):
        wav, target = _wav_target()
        stretch = FrameTimeStretch()

        with patch("musicality.augmentations.random.uniform", return_value=1.5):
            new_wav, new_target, _ = stretch(wav, target, SR)

        assert new_wav.shape[-1] < wav.shape[-1]
        assert new_target.shape[-1] < target.shape[-1]

    def test_rate_less_than_one_grows_length(self):
        wav, target = _wav_target()
        stretch = FrameTimeStretch()

        with patch("musicality.augmentations.random.uniform", return_value=0.7):
            new_wav, new_target, _ = stretch(wav, target, SR)

        assert new_wav.shape[-1] > wav.shape[-1]
        assert new_target.shape[-1] > target.shape[-1]

    def test_target_channel_count_preserved(self):
        wav, target = _wav_target()
        stretch = FrameTimeStretch()

        with patch("musicality.augmentations.random.uniform", return_value=1.1):
            _, new_target, _ = stretch(wav, target, SR)

        assert new_target.shape[0] == 4

    @pytest.mark.parametrize("rate", [0.85, 0.93, 1.07, 1.15])
    def test_moved_beats_follow_the_stretched_peaks(self, rate):
        wav, target = _wav_target()
        beats = _beats()

        with patch("musicality.augmentations.random.uniform", return_value=rate):
            _, new_target, new_beats = FrameTimeStretch()(wav, target, SR, beats)

        moved = new_beats[0, new_beats[0] >= 0]
        peaks = _peaks(new_target[0])

        assert len(moved) == len(peaks) == 20
        assert torch.all((moved - peaks).abs() <= 1.0)
        assert torch.equal(new_beats[1], beats[1])

    def test_padding_is_left_alone(self):
        wav, target = _wav_target()
        beats = _beats()

        with patch("musicality.augmentations.random.uniform", return_value=0.85):
            _, _, new_beats = FrameTimeStretch()(wav, target, SR, beats)

        assert torch.all(new_beats[0, 20:] == PAD_TIME)
        assert torch.all(new_beats[1, 20:] == 0)

    def test_beats_none_stays_none(self):
        wav, target = _wav_target()

        _, _, beats = FrameTimeStretch()(wav, target, SR)

        assert beats is None


# ---------------------------------------------------------------------------
# BeatPhaseAugmenter
# ---------------------------------------------------------------------------


class TestBeatPhaseAugmenter:
    def test_no_augmentations_returns_shapes_unchanged(self):
        wav, target = _wav_target()
        augmenter = BeatPhaseAugmenter()
        new_wav, new_target, _ = augmenter(wav, target, SR, N_SAMPLES, N_FRAMES)

        assert torch.equal(new_wav, wav)
        assert torch.equal(new_target, target)

    def test_time_stretch_output_cropped_or_padded_to_fixed_lengths(self):
        wav, target = _wav_target()
        augmenter = BeatPhaseAugmenter(
            time_stretch=FrameTimeStretch(min_rate=0.7, max_rate=1.5)
        )

        for _ in range(10):  # several random rates, both shrink and grow cases
            new_wav, new_target, _ = augmenter(wav, target, SR, N_SAMPLES, N_FRAMES)
            assert new_wav.shape == (1, N_SAMPLES)
            assert new_target.shape == (4, N_FRAMES)

    def test_gain_changes_wav_not_target(self):
        wav, target = _wav_target()
        augmenter = BeatPhaseAugmenter(gain=RandomGain(min_db=6.0, max_db=6.0))
        new_wav, new_target, _ = augmenter(wav, target, SR, N_SAMPLES, N_FRAMES)

        assert not torch.equal(new_wav, wav)
        assert torch.equal(new_target, target)

    def test_noise_changes_wav_not_target(self):
        wav, target = _wav_target()
        augmenter = BeatPhaseAugmenter(noise=AddNoise(std=0.1))
        new_wav, new_target, _ = augmenter(wav, target, SR, N_SAMPLES, N_FRAMES)

        assert not torch.equal(new_wav, wav)
        assert torch.equal(new_target, target)

    def test_beats_stretched_past_the_clip_become_padding(self):
        """At rate 0.8 a beat at frame b moves to (b + 0.5) * 1.25 - 0.5, so
        the 16 beats at frames 0..150 stay inside 200 frames and the 4 after
        them are pushed out."""
        wav, target = _wav_target()
        augmenter = BeatPhaseAugmenter(time_stretch=FrameTimeStretch())

        with patch("musicality.augmentations.random.uniform", return_value=0.8):
            _, _, beats = augmenter(wav, target, SR, N_SAMPLES, N_FRAMES, _beats())

        valid = beats[0] >= 0

        assert torch.equal(valid[:16], torch.ones(16, dtype=torch.bool))
        assert not valid[16:].any()
        assert torch.all(beats[0, valid].round() < N_FRAMES)
        assert torch.all(beats[1, 16:] == 0)


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

    def test_forwards_beats_only_when_the_subset_yields_them(self):
        augmenter = BeatPhaseAugmenter(time_stretch=FrameTimeStretch())

        pairs = AugmentedBeatDataset(
            [_wav_target()], augmenter, SR, N_SAMPLES, N_FRAMES
        )
        triples = AugmentedBeatDataset(
            [(*_wav_target(), _beats())], augmenter, SR, N_SAMPLES, N_FRAMES
        )

        assert len(pairs[0]) == 2
        assert triples[0][2].shape == (2, MAX_BEATS)


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
