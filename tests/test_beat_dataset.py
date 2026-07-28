"""Tests for musicality.loaders.beat_dataset — uses mocks to avoid disk I/O."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
import pytest

SAMPLE_RATE = 22050
DURATION = 5.0
HOP_LENGTH = 512
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_FRAMES = N_SAMPLES // HOP_LENGTH

BEAT, ONE, FOUR, MASK = range(4)


def _make_track(tid="t1", beat_times=None, positions=None):
    """Build a fake mirdata track object with the given beat timestamps (in seconds)
    and optional bar positions (1-indexed, ``bar_index`` unit).

    If beat_times is None, the track has no beat annotation (simulates an
    unannotated track that the dataset should skip). If positions is None but
    beat_times is given, the track has beats but no bar-position annotation.
    """
    t = MagicMock()
    t.track_id = tid
    t.audio_path = f"/fake/{tid}.wav"
    if beat_times is not None:
        t.beats = MagicMock()
        t.beats.times = np.array(beat_times)
        t.beats.positions = np.array(positions) if positions is not None else None
    else:
        t.beats = None
    return t


def _fake_ds(tracks):
    """Build a fake mirdata dataset whose track_ids and track() match the given list."""
    ds = MagicMock()
    ds.track_ids = [t.track_id for t in tracks]
    ds.track.side_effect = {t.track_id: t for t in tracks}.__getitem__
    return ds


def _patch_loader(tracks, wav_shape=(1, N_SAMPLES), sr=SAMPLE_RATE):
    """Return three context managers that replace external I/O with fakes:
    - mirdata.initialize → returns a fake dataset built from tracks
    - torchaudio.load    → returns a random tensor of wav_shape at sample rate sr
    - Path.exists        → always returns True (no real files needed)
    """
    fake_ds = _fake_ds(tracks)
    fake_wav = torch.randn(*wav_shape)
    return (
        patch(
            "musicality.loaders.beat_dataset.mirdata.initialize", return_value=fake_ds
        ),
        patch(
            "musicality.loaders.beat_dataset.torchaudio.load",
            return_value=(fake_wav, sr),
        ),
        patch("pathlib.Path.exists", return_value=True),
    )


class TestBeatDataset:
    def test_len(self):
        """Dataset length equals the number of tracks that have beat annotations."""
        tracks = [
            _make_track(f"t{i}", beat_times=[0.5 * j for j in range(8)])
            for i in range(4)
        ]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION
            )
        assert len(ds) == 4

    def test_skips_missing_beats(self):
        """Tracks with no beat annotation (beats=None) are silently excluded."""
        tracks = [
            _make_track("t1", beat_times=[0.5, 1.0]),
            _make_track("t2", beat_times=None),
        ]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION
            )
        assert len(ds) == 1

    def test_output_shapes(self):
        """Each item is a (waveform, target) pair with the expected fixed shapes.

        The waveform is (1, N_SAMPLES) — mono, fixed length.
        The target is (4, N_FRAMES) — beat/one/four/mask channels, one value
        per hop-length frame.
        """
        tracks = [_make_track("t1", beat_times=[0.5, 1.0, 1.5], positions=[1, 2, 3])]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="ballroom",
                sample_rate=SAMPLE_RATE,
                duration=DURATION,
                hop_length=HOP_LENGTH,
            )
            wav, target = ds[0]

        assert wav.shape == (1, N_SAMPLES)
        assert target.shape == (4, N_FRAMES)

    def test_beat_frames_are_set(self):
        """The beat-channel frame corresponding to each beat timestamp must peak at 1.0.

        Beat time t maps to frame index round(t * sample_rate / hop_length).
        """
        beat_times = [0.5, 1.0]
        tracks = [_make_track("t1", beat_times=beat_times)]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="ballroom",
                sample_rate=SAMPLE_RATE,
                duration=DURATION,
                hop_length=HOP_LENGTH,
            )
            _, target = ds[0]

        for t in beat_times:
            frame = round(t * SAMPLE_RATE / HOP_LENGTH)
            assert target[BEAT, frame].item() == pytest.approx(1.0)

    def test_target_in_unit_range(self):
        """Every value in the smeared target tensor lies in [0, 1]."""
        tracks = [
            _make_track("t1", beat_times=[0.5, 1.0, 1.5, 2.0], positions=[1, 2, 3, 4])
        ]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION
            )
            _, target = ds[0]

        assert target.min().item() >= 0.0
        assert target.max().item() <= 1.0

    def test_beat_outside_clip_ignored(self):
        """A beat that falls after the clip duration must not appear in the target."""
        tracks = [_make_track("t1", beat_times=[DURATION + 1.0])]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION
            )
            _, target = ds[0]

        assert target[BEAT].sum().item() == pytest.approx(0.0)

    def test_one_and_four_channels_from_positions(self):
        """Bar positions 1 and 4 populate the one/four channels at the right frames."""
        beat_times = [0.5, 1.0, 1.5, 2.0, 2.5]
        positions = [1, 2, 3, 4, 1]
        tracks = [_make_track("t1", beat_times=beat_times, positions=positions)]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="ballroom",
                sample_rate=SAMPLE_RATE,
                duration=DURATION,
                hop_length=HOP_LENGTH,
            )
            _, target = ds[0]

        one_frames = [
            round(t * SAMPLE_RATE / HOP_LENGTH)
            for t, p in zip(beat_times, positions)
            if p == 1
        ]
        four_frames = [
            round(t * SAMPLE_RATE / HOP_LENGTH)
            for t, p in zip(beat_times, positions)
            if p == 4
        ]

        for frame in one_frames:
            assert target[ONE, frame].item() == pytest.approx(1.0)
        for frame in four_frames:
            assert target[FOUR, frame].item() == pytest.approx(1.0)

        # Frame at beat position 2 (1.0s) must not register in either channel.
        mid_frame = round(1.0 * SAMPLE_RATE / HOP_LENGTH)
        assert target[ONE, mid_frame].item() == pytest.approx(0.0)
        assert target[FOUR, mid_frame].item() == pytest.approx(0.0)

    def test_mask_set_when_positions_present(self):
        """The mask channel is constant 1.0 when the track has bar-position annotations."""
        tracks = [_make_track("t1", beat_times=[0.5, 1.0], positions=[1, 2])]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION
            )
            _, target = ds[0]

        assert torch.all(target[MASK] == 1.0)

    def test_mask_unset_when_positions_missing(self):
        """The mask channel is constant 0.0, and one/four stay empty, without position annotations."""
        tracks = [_make_track("t1", beat_times=[0.5, 1.0], positions=None)]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="rwc_popular", sample_rate=SAMPLE_RATE, duration=DURATION
            )
            _, target = ds[0]

        assert torch.all(target[MASK] == 0.0)
        assert torch.all(target[ONE] == 0.0)
        assert torch.all(target[FOUR] == 0.0)


class TestGaussianSmear:
    def test_isolated_spike_peak_is_one(self):
        from musicality.loaders.beat_dataset import gaussian_smear

        spike = np.zeros(50, dtype=np.float32)
        spike[25] = 1.0
        smeared = gaussian_smear(spike, sigma=1.5)

        assert smeared[25] == pytest.approx(1.0)
        assert smeared.max() == pytest.approx(1.0)

    def test_smear_spreads_to_neighbors(self):
        from musicality.loaders.beat_dataset import gaussian_smear

        spike = np.zeros(50, dtype=np.float32)
        spike[25] = 1.0
        smeared = gaussian_smear(spike, sigma=1.5)

        assert 0.0 < smeared[24] < 1.0
        assert 0.0 < smeared[26] < 1.0

    def test_no_spike_is_all_zero(self):
        from musicality.loaders.beat_dataset import gaussian_smear

        spike = np.zeros(50, dtype=np.float32)
        smeared = gaussian_smear(spike, sigma=1.5)

        assert np.all(smeared == 0.0)

    def test_clips_overlapping_bumps_to_one(self):
        from musicality.loaders.beat_dataset import gaussian_smear

        spike = np.zeros(50, dtype=np.float32)
        spike[25] = 1.0
        spike[26] = 1.0
        smeared = gaussian_smear(spike, sigma=1.5)

        assert smeared.max() == pytest.approx(1.0)
        assert smeared.max() <= 1.0


class TestDataLoader:
    def test_batch_shape(self):
        from torch.utils.data import DataLoader
        from musicality.loaders.beat_dataset import BeatDataset

        tracks = [
            _make_track(
                f"t{i}", beat_times=[0.5 * j for j in range(4)], positions=[1, 2, 3, 4]
            )
            for i in range(4)
        ]
        patches = _patch_loader(tracks)
        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(
                name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION
            )
            loader = DataLoader(ds, batch_size=4, shuffle=False)
            wav, target = next(iter(loader))

        assert wav.shape == (4, 1, N_SAMPLES)
        assert target.shape == (4, 4, N_FRAMES)
