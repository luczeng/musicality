"""Tests for musicality.loaders.tempo_dataset — real tracks/+annotations/
fixtures on disk (torchaudio.load is mocked, since audio content is
irrelevant to this loader's own logic — only bpm_median from .meta.json and
file presence/absence matter here).
"""

import json
from unittest.mock import patch

import pytest
import torch

SAMPLE_RATE = 22050
N_SAMPLES = SAMPLE_RATE * 10  # 10 s


def _write_track(dataset_dir, track_id, bpm_median=None, has_metadata=True):
    """Create tracks/<id>.wav + annotations/<id>.beats (+.meta.json) for one
    track. *has_metadata=False* omits the .meta.json file entirely (simulates
    a migrated dataset that hasn't had bpm_stats computed for this track);
    *bpm_median=None* with *has_metadata=True* writes metadata with no tempo.
    """
    tracks_dir = dataset_dir / "tracks"
    ann_dir = dataset_dir / "annotations"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    (tracks_dir / f"{track_id}.wav").write_bytes(b"")
    (ann_dir / f"{track_id}.beats").write_text("0.5 1\n1.0 2\n")
    if has_metadata:
        (ann_dir / f"{track_id}.meta.json").write_text(
            json.dumps({"bpm_median": bpm_median})
        )


def _patch_audio(wav_shape=(2, N_SAMPLES), sr=SAMPLE_RATE):
    fake_wav = torch.randn(*wav_shape)
    return patch(
        "musicality.loaders.tempo_dataset.torchaudio.load", return_value=(fake_wav, sr)
    )


# ---------------------------------------------------------------------------
# TempoDataset
# ---------------------------------------------------------------------------


class TestTempoDataset:
    def test_len(self, tmp_path):
        dataset_dir = tmp_path / "brid"
        for i in range(5):
            _write_track(dataset_dir, f"t{i}", bpm_median=100 + i)
        from musicality.loaders.tempo_dataset import TempoDataset

        with _patch_audio():
            ds = TempoDataset(name="brid", data_home=dataset_dir)
        assert len(ds) == 5

    def test_skips_none_tempo(self, tmp_path):
        dataset_dir = tmp_path / "brid"
        _write_track(dataset_dir, "t1", bpm_median=120.0)
        _write_track(dataset_dir, "t2", bpm_median=None)
        from musicality.loaders.tempo_dataset import TempoDataset

        with _patch_audio():
            ds = TempoDataset(name="brid", data_home=dataset_dir)
        assert len(ds) == 1

    def test_skips_missing_metadata(self, tmp_path):
        """A migrated .beats file with no .meta.json at all is skipped too."""
        dataset_dir = tmp_path / "brid"
        _write_track(dataset_dir, "t1", bpm_median=120.0)
        _write_track(dataset_dir, "t2", has_metadata=False)
        from musicality.loaders.tempo_dataset import TempoDataset

        with _patch_audio():
            ds = TempoDataset(name="brid", data_home=dataset_dir)
        assert len(ds) == 1

    def test_item_shapes(self, tmp_path):
        dataset_dir = tmp_path / "brid"
        _write_track(dataset_dir, "t1", bpm_median=128.0)
        from musicality.loaders.tempo_dataset import TempoDataset

        with _patch_audio():
            ds = TempoDataset(name="brid", data_home=dataset_dir)
            wav, label = ds[0]

        assert wav.ndim == 2  # (1, T)
        assert wav.shape[0] == 1
        assert label.shape == ()
        assert label.item() == pytest.approx(128.0)

    def test_mono_mixdown(self, tmp_path):
        """Stereo input (2, N) must produce a (1, T) waveform."""
        dataset_dir = tmp_path / "brid"
        _write_track(dataset_dir, "t1", bpm_median=100.0)
        from musicality.loaders.tempo_dataset import TempoDataset

        with _patch_audio(wav_shape=(2, N_SAMPLES)):
            ds = TempoDataset(name="brid", data_home=dataset_dir)
            wav, _ = ds[0]
        assert wav.shape[0] == 1

    def test_truncation(self, tmp_path):
        """Audio longer or shorter than duration produces the same fixed output shape."""
        dataset_dir = tmp_path / "brid"
        _write_track(dataset_dir, "t1", bpm_median=90.0)
        from musicality.loaders.tempo_dataset import TempoDataset

        with _patch_audio(wav_shape=(1, N_SAMPLES * 2)):
            ds = TempoDataset(name="brid", data_home=dataset_dir)
            wav1, _ = ds[0]

        with _patch_audio(wav_shape=(1, N_SAMPLES // 2)):
            ds2 = TempoDataset(name="brid", data_home=dataset_dir)
            wav2, _ = ds2[0]

        assert wav1.shape == wav2.shape

    def test_resample(self, tmp_path):
        """Audio at wrong sample rate is resampled; output shape stays fixed."""
        dataset_dir = tmp_path / "brid"
        _write_track(dataset_dir, "t1", bpm_median=100.0)
        wrong_sr = 44100
        n_at_wrong_sr = int(10.0 * wrong_sr)
        from musicality.loaders.tempo_dataset import TempoDataset

        with _patch_audio(wav_shape=(1, n_at_wrong_sr), sr=wrong_sr):
            ds = TempoDataset(name="brid", data_home=dataset_dir, sample_rate=22050)
            wav, _ = ds[0]
        assert wav.shape[0] == 1


# ---------------------------------------------------------------------------
# refs= construction (tracks pulled from multiple source datasets)
# ---------------------------------------------------------------------------


class TestRefsConstruction:
    def test_pulls_samples_from_multiple_source_datasets(self, tmp_path):
        _write_track(tmp_path / "ballroom", "a", bpm_median=120.0)
        _write_track(tmp_path / "brid", "b", bpm_median=90.0)

        from musicality.dataformats.track_io import TrackRef
        from musicality.loaders.tempo_dataset import TempoDataset

        refs = [
            TrackRef("ballroom", "a", tmp_path / "ballroom"),
            TrackRef("brid", "b", tmp_path / "brid"),
        ]

        with _patch_audio():
            ds = TempoDataset(refs=refs)

        assert len(ds) == 2
        audio_paths = {path for path, _ in ds.samples}
        assert any("ballroom" in p for p in audio_paths)
        assert any("brid" in p for p in audio_paths)
        assert sorted(tempo for _, tempo in ds.samples) == [90.0, 120.0]

    def test_skips_unresolvable_metadata_across_sources(self, tmp_path):
        _write_track(tmp_path / "ballroom", "a", bpm_median=120.0)
        _write_track(tmp_path / "brid", "b", bpm_median=None)

        from musicality.dataformats.track_io import TrackRef
        from musicality.loaders.tempo_dataset import TempoDataset

        refs = [
            TrackRef("ballroom", "a", tmp_path / "ballroom"),
            TrackRef("brid", "b", tmp_path / "brid"),
        ]

        with _patch_audio():
            ds = TempoDataset(refs=refs)

        assert len(ds) == 1

    def test_refs_populated_in_lockstep_with_samples(self, tmp_path):
        _write_track(tmp_path / "ballroom", "a", bpm_median=120.0)
        _write_track(tmp_path / "brid", "b", bpm_median=None)

        from musicality.dataformats.track_io import TrackRef
        from musicality.loaders.tempo_dataset import TempoDataset

        refs = [
            TrackRef("ballroom", "a", tmp_path / "ballroom"),
            TrackRef("brid", "b", tmp_path / "brid"),
        ]

        with _patch_audio():
            ds = TempoDataset(refs=refs)

        assert len(ds.refs) == len(ds.samples) == 1
        assert ds.refs[0].dataset_name == "ballroom"
        assert ds.refs[0].track_id == "a"

    def test_requires_name_or_refs(self):
        from musicality.loaders.tempo_dataset import TempoDataset

        with pytest.raises(ValueError):
            TempoDataset()


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_returns_dataloader(self, tmp_path):
        from torch.utils.data import DataLoader
        from musicality.loaders.tempo_dataset import TempoDataset

        dataset_dir = tmp_path / "brid"
        for i in range(8):
            _write_track(dataset_dir, f"t{i}", bpm_median=float(i * 10 + 60))

        with _patch_audio():
            ds = TempoDataset(name="brid", data_home=dataset_dir)
            loader = DataLoader(ds, batch_size=4)
        assert isinstance(loader, DataLoader)

    def test_batch_shape(self, tmp_path):
        from torch.utils.data import DataLoader
        from musicality.loaders.tempo_dataset import TempoDataset

        dataset_dir = tmp_path / "brid"
        for i in range(8):
            _write_track(dataset_dir, f"t{i}", bpm_median=float(i * 10 + 60))

        with _patch_audio():
            ds = TempoDataset(name="brid", data_home=dataset_dir)
            loader = DataLoader(ds, batch_size=4, shuffle=False)
            wav, tempo = next(iter(loader))
        assert wav.shape[0] == 4
        assert wav.shape[1] == 1
        assert tempo.shape == (4,)

    def test_tempo_values(self, tmp_path):
        from torch.utils.data import DataLoader
        from musicality.loaders.tempo_dataset import TempoDataset

        tempos = [60.0, 90.0, 120.0, 150.0]
        dataset_dir = tmp_path / "brid"
        for i, t in enumerate(tempos):
            _write_track(dataset_dir, f"t{i}", bpm_median=t)

        with _patch_audio():
            ds = TempoDataset(name="brid", data_home=dataset_dir)
            loader = DataLoader(ds, batch_size=4, shuffle=False)
            _, tempo = next(iter(loader))
        assert sorted(tempo.tolist()) == pytest.approx(sorted(tempos))
