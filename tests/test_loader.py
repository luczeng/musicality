"""Tests for musicality.loader — uses mocks to avoid disk I/O."""

from unittest.mock import MagicMock, patch

import torch
import pytest

SAMPLE_RATE = 22050
N_SAMPLES = SAMPLE_RATE * 10  # 10 s


def _make_track(tid="t1", tempo=120.0, exists=True):
    t = MagicMock()
    t.track_id = tid
    t.tempo = tempo
    t.audio_path = f"/fake/{tid}.wav"
    return t


def _fake_ds(tracks):
    ds = MagicMock()
    ds.track_ids = [t.track_id for t in tracks]
    ds.track.side_effect = {t.track_id: t for t in tracks}.__getitem__
    return ds


def _patch_loader(tracks, wav_shape=(2, N_SAMPLES), sr=SAMPLE_RATE):
    """Context managers that patch mirdata and torchaudio for BRIDDataset."""
    fake_ds = _fake_ds(tracks)
    fake_wav = torch.randn(*wav_shape)

    mirdata_patch = patch("musicality.loader.mirdata.initialize", return_value=fake_ds)
    audio_patch = patch("musicality.loader.torchaudio.load", return_value=(fake_wav, sr))
    exists_patch = patch("os.path.exists", return_value=True)
    return mirdata_patch, audio_patch, exists_patch


# ---------------------------------------------------------------------------
# BRIDDataset
# ---------------------------------------------------------------------------

class TestBRIDDataset:
    def _make(self, tracks, **kwargs):
        from musicality.loader import BRIDDataset
        patches = _patch_loader(tracks)
        with patches[0], patches[1], patches[2]:
            return BRIDDataset(**kwargs), patches

    def test_len(self):
        tracks = [_make_track(f"t{i}", tempo=100 + i) for i in range(5)]
        patches = _patch_loader(tracks)
        from musicality.loader import BRIDDataset
        with patches[0], patches[1], patches[2]:
            ds = BRIDDataset()
        assert len(ds) == 5

    def test_skips_none_tempo(self):
        tracks = [_make_track("t1", tempo=120.0), _make_track("t2", tempo=None)]
        patches = _patch_loader(tracks)
        from musicality.loader import BRIDDataset
        with patches[0], patches[1], patches[2]:
            ds = BRIDDataset()
        assert len(ds) == 1

    def test_item_shapes(self):
        tracks = [_make_track("t1", tempo=128.0)]
        patches = _patch_loader(tracks)
        from musicality.loader import BRIDDataset
        with patches[0], patches[1], patches[2]:
            ds = BRIDDataset(n_mels=64)
            mel, label = ds[0]

        assert mel.ndim == 3          # (1, n_mels, T)
        assert mel.shape[0] == 1
        assert mel.shape[1] == 64
        assert label.shape == ()
        assert label.item() == pytest.approx(128.0)

    def test_mono_mixdown(self):
        """Stereo input (2, N) must produce a (1, n_mels, T) spectrogram."""
        tracks = [_make_track("t1", tempo=100.0)]
        patches = _patch_loader(tracks, wav_shape=(2, N_SAMPLES))
        from musicality.loader import BRIDDataset
        with patches[0], patches[1], patches[2]:
            ds = BRIDDataset()
            mel, _ = ds[0]
        assert mel.shape[0] == 1

    def test_truncation(self):
        """Audio longer than duration is truncated — output time dim is fixed."""
        tracks = [_make_track("t1", tempo=90.0)]
        patches = _patch_loader(tracks, wav_shape=(1, N_SAMPLES * 2))
        from musicality.loader import BRIDDataset
        with patches[0], patches[1], patches[2]:
            ds = BRIDDataset()
            mel1, _ = ds[0]

        patches2 = _patch_loader(tracks, wav_shape=(1, N_SAMPLES // 2))
        with patches2[0], patches2[1], patches2[2]:
            ds2 = BRIDDataset()
            mel2, _ = ds2[0]

        assert mel1.shape == mel2.shape  # same fixed output size regardless of input length

    def test_resample(self):
        """Audio at wrong sample rate is resampled; output shape stays fixed."""
        tracks = [_make_track("t1", tempo=100.0)]
        wrong_sr = 44100
        n_at_wrong_sr = int(10.0 * wrong_sr)
        patches = _patch_loader(tracks, wav_shape=(1, n_at_wrong_sr), sr=wrong_sr)
        from musicality.loader import BRIDDataset
        with patches[0], patches[1], patches[2]:
            ds = BRIDDataset(sample_rate=22050)
            mel, _ = ds[0]
        assert mel.shape[0] == 1


# ---------------------------------------------------------------------------
# get_loader
# ---------------------------------------------------------------------------

class TestGetLoader:
    def test_returns_dataloader(self):
        from torch.utils.data import DataLoader
        from musicality.loader import get_loader
        tracks = [_make_track(f"t{i}", tempo=float(i * 10 + 60)) for i in range(8)]
        patches = _patch_loader(tracks)
        with patches[0], patches[1], patches[2]:
            loader = get_loader(batch_size=4)
        assert isinstance(loader, DataLoader)

    def test_batch_shape(self):
        from musicality.loader import get_loader
        tracks = [_make_track(f"t{i}", tempo=float(i * 10 + 60)) for i in range(8)]
        patches = _patch_loader(tracks)
        with patches[0], patches[1], patches[2]:
            loader = get_loader(batch_size=4, shuffle=False, n_mels=64)
            mel, tempo = next(iter(loader))
        assert mel.shape[0] == 4
        assert mel.shape[1] == 1
        assert mel.shape[2] == 64
        assert tempo.shape == (4,)

    def test_tempo_values(self):
        tempos = [60.0, 90.0, 120.0, 150.0]
        tracks = [_make_track(f"t{i}", tempo=t) for i, t in enumerate(tempos)]
        patches = _patch_loader(tracks)
        from musicality.loader import get_loader
        with patches[0], patches[1], patches[2]:
            loader = get_loader(batch_size=4, shuffle=False)
            _, tempo = next(iter(loader))
        assert sorted(tempo.tolist()) == pytest.approx(sorted(tempos))
