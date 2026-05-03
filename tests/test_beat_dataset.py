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


def _make_track(tid="t1", beat_times=None):
    """Build a fake mirdata track object with the given beat timestamps (in seconds).

    If beat_times is None, the track has no beat annotation (simulates an
    unannotated track that the dataset should skip).
    """
    t = MagicMock()
    t.track_id = tid
    t.audio_path = f"/fake/{tid}.wav"
    if beat_times is not None:
        t.beats = MagicMock()
        t.beats.times = np.array(beat_times)
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
        patch("musicality.loaders.beat_dataset.mirdata.initialize", return_value=fake_ds),
        patch("musicality.loaders.beat_dataset.torchaudio.load", return_value=(fake_wav, sr)),
        patch("pathlib.Path.exists", return_value=True),
    )


class TestBeatDataset:
    def test_len(self):
        """Dataset length equals the number of tracks that have beat annotations."""
        tracks = [_make_track(f"t{i}", beat_times=[0.5 * j for j in range(8)]) for i in range(4)]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION)
        assert len(ds) == 4

    def test_skips_missing_beats(self):
        """Tracks with no beat annotation (beats=None) are silently excluded."""
        tracks = [_make_track("t1", beat_times=[0.5, 1.0]), _make_track("t2", beat_times=None)]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION)
        assert len(ds) == 1

    def test_output_shapes(self):
        """Each item is a (waveform, target) pair with the expected fixed shapes.

        The waveform is (1, N_SAMPLES) — mono, fixed length.
        The target is (N_FRAMES,) — one value per hop-length frame.
        """
        tracks = [_make_track("t1", beat_times=[0.5, 1.0, 1.5])]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION, hop_length=HOP_LENGTH)
            wav, target = ds[0]

        assert wav.shape == (1, N_SAMPLES)
        assert target.shape == (N_FRAMES,)

    def test_beat_frames_are_set(self):
        """The target frame corresponding to each beat timestamp must be 1.0.

        Beat time t maps to frame index round(t * sample_rate / hop_length).
        """
        beat_times = [0.5, 1.0]
        tracks = [_make_track("t1", beat_times=beat_times)]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION, hop_length=HOP_LENGTH)
            _, target = ds[0]

        for t in beat_times:
            frame = round(t * SAMPLE_RATE / HOP_LENGTH)
            assert target[frame].item() == pytest.approx(1.0)

    def test_target_is_binary(self):
        """Every value in the target tensor is either 0.0 (no beat) or 1.0 (beat)."""
        tracks = [_make_track("t1", beat_times=[0.5, 1.0, 1.5, 2.0])]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION)
            _, target = ds[0]

        assert set(target.unique().tolist()).issubset({0.0, 1.0})

    def test_beat_outside_clip_ignored(self):
        """A beat that falls after the clip duration must not appear in the target."""
        tracks = [_make_track("t1", beat_times=[DURATION + 1.0])]
        patches = _patch_loader(tracks)
        from musicality.loaders.beat_dataset import BeatDataset

        with patches[0], patches[1], patches[2]:
            ds = BeatDataset(name="ballroom", sample_rate=SAMPLE_RATE, duration=DURATION)
            _, target = ds[0]

        assert target.sum().item() == pytest.approx(0.0)
