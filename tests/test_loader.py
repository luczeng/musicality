"""Tests for musicality.loaders.loader — uses mocks to avoid disk I/O."""

from unittest.mock import MagicMock, patch

import torch
import pytest

SAMPLE_RATE = 22050
N_SAMPLES = SAMPLE_RATE * 10  # 10 s


def _make_track(tid="t1", tempo=120.0):
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
    """Context managers that patch mirdata and torchaudio for TempoDataset."""
    fake_ds = _fake_ds(tracks)
    fake_wav = torch.randn(*wav_shape)

    mirdata_patch = patch(
        "musicality.loaders.loader.mirdata.initialize", return_value=fake_ds
    )
    audio_patch = patch(
        "musicality.loaders.loader.torchaudio.load", return_value=(fake_wav, sr)
    )
    exists_patch = patch("pathlib.Path.exists", return_value=True)
    return mirdata_patch, audio_patch, exists_patch


# ---------------------------------------------------------------------------
# TempoDataset
# ---------------------------------------------------------------------------


class TestTempoDataset:
    def test_len(self):
        tracks = [_make_track(f"t{i}", tempo=100 + i) for i in range(5)]
        patches = _patch_loader(tracks)
        from musicality.loaders.loader import TempoDataset

        with patches[0], patches[1], patches[2]:
            ds = TempoDataset(name="brid")
        assert len(ds) == 5

    def test_skips_none_tempo(self):
        tracks = [_make_track("t1", tempo=120.0), _make_track("t2", tempo=None)]
        patches = _patch_loader(tracks)
        from musicality.loaders.loader import TempoDataset

        with patches[0], patches[1], patches[2]:
            ds = TempoDataset(name="brid")
        assert len(ds) == 1

    def test_item_shapes(self):
        tracks = [_make_track("t1", tempo=128.0)]
        patches = _patch_loader(tracks)
        from musicality.loaders.loader import TempoDataset

        with patches[0], patches[1], patches[2]:
            ds = TempoDataset(name="brid")
            wav, label = ds[0]

        assert wav.ndim == 2  # (1, T)
        assert wav.shape[0] == 1
        assert label.shape == ()
        assert label.item() == pytest.approx(128.0)

    def test_mono_mixdown(self):
        """Stereo input (2, N) must produce a (1, T) waveform."""
        tracks = [_make_track("t1", tempo=100.0)]
        patches = _patch_loader(tracks, wav_shape=(2, N_SAMPLES))
        from musicality.loaders.loader import TempoDataset

        with patches[0], patches[1], patches[2]:
            ds = TempoDataset(name="brid")
            wav, _ = ds[0]
        assert wav.shape[0] == 1

    def test_truncation(self):
        """Audio longer or shorter than duration produces the same fixed output shape."""
        tracks = [_make_track("t1", tempo=90.0)]
        patches = _patch_loader(tracks, wav_shape=(1, N_SAMPLES * 2))
        from musicality.loaders.loader import TempoDataset

        with patches[0], patches[1], patches[2]:
            ds = TempoDataset(name="brid")
            wav1, _ = ds[0]

        patches2 = _patch_loader(tracks, wav_shape=(1, N_SAMPLES // 2))
        with patches2[0], patches2[1], patches2[2]:
            ds2 = TempoDataset(name="brid")
            wav2, _ = ds2[0]

        assert wav1.shape == wav2.shape

    def test_resample(self):
        """Audio at wrong sample rate is resampled; output shape stays fixed."""
        tracks = [_make_track("t1", tempo=100.0)]
        wrong_sr = 44100
        n_at_wrong_sr = int(10.0 * wrong_sr)
        patches = _patch_loader(tracks, wav_shape=(1, n_at_wrong_sr), sr=wrong_sr)
        from musicality.loaders.loader import TempoDataset

        with patches[0], patches[1], patches[2]:
            ds = TempoDataset(name="brid", sample_rate=22050)
            wav, _ = ds[0]
        assert wav.shape[0] == 1


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_returns_dataloader(self):
        from torch.utils.data import DataLoader
        from musicality.loaders.loader import TempoDataset

        tracks = [_make_track(f"t{i}", tempo=float(i * 10 + 60)) for i in range(8)]
        patches = _patch_loader(tracks)
        with patches[0], patches[1], patches[2]:
            ds = TempoDataset(name="brid")
            loader = DataLoader(ds, batch_size=4)
        assert isinstance(loader, DataLoader)

    def test_batch_shape(self):
        from torch.utils.data import DataLoader
        from musicality.loaders.loader import TempoDataset

        tracks = [_make_track(f"t{i}", tempo=float(i * 10 + 60)) for i in range(8)]
        patches = _patch_loader(tracks)
        with patches[0], patches[1], patches[2]:
            ds = TempoDataset(name="brid")
            loader = DataLoader(ds, batch_size=4, shuffle=False)
            wav, tempo = next(iter(loader))
        assert wav.shape[0] == 4
        assert wav.shape[1] == 1
        assert tempo.shape == (4,)

    def test_tempo_values(self):
        tempos = [60.0, 90.0, 120.0, 150.0]
        tracks = [_make_track(f"t{i}", tempo=t) for i, t in enumerate(tempos)]
        patches = _patch_loader(tracks)
        from torch.utils.data import DataLoader
        from musicality.loaders.loader import TempoDataset

        with patches[0], patches[1], patches[2]:
            ds = TempoDataset(name="brid")
            loader = DataLoader(ds, batch_size=4, shuffle=False)
            _, tempo = next(iter(loader))
        assert sorted(tempo.tolist()) == pytest.approx(sorted(tempos))
