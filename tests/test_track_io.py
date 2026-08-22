"""Tests for musicality.dataformats.track_io's new lookups introduced when
this module was split out of tools/annotator/data.py.

TrackData/TrackMetadata, read_beats_file, save_annotations/save_metadata/
load_metadata, and annotation_path/metadata_path are moved-as-is code
already covered by tests/test_annotator_data.py and
tests/test_annotator_metadata.py, which exercise the same code through
tools.annotator.data's re-export. This file covers only what's genuinely
new: list_migrated_track_ids, resolve_track_audio, and bpm_stats.
"""

import numpy as np
import pytest

import musicality.dataformats as dataformats
from musicality.dataformats.track_io import (
    bpm_stats,
    list_migrated_track_ids,
    resolve_track_audio,
)


# ---------------------------------------------------------------------------
# list_migrated_track_ids
# ---------------------------------------------------------------------------


class TestListMigratedTrackIds:
    def test_no_annotations_dir_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path)
        assert list_migrated_track_ids("nope") == []

    def test_empty_annotations_dir_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path)
        (tmp_path / "swing" / "annotations").mkdir(parents=True)
        assert list_migrated_track_ids("swing") == []

    def test_lists_beats_file_stems_sorted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path)
        ann_dir = tmp_path / "swing" / "annotations"
        ann_dir.mkdir(parents=True)
        (ann_dir / "b.beats").write_text("1.0 1")
        (ann_dir / "a.beats").write_text("1.0 1")
        assert list_migrated_track_ids("swing") == ["a", "b"]

    def test_ignores_non_beats_files(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path)
        ann_dir = tmp_path / "swing" / "annotations"
        ann_dir.mkdir(parents=True)
        (ann_dir / "a.beats").write_text("1.0 1")
        (ann_dir / "a.meta.json").write_text("{}")
        assert list_migrated_track_ids("swing") == ["a"]

    def test_excludes_annotator_subdirectories(self, monkeypatch, tmp_path):
        """Only the default (top-level) annotation slot counts as migrated —
        an alternate annotator's subdirectory isn't migration output."""
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path)
        ann_dir = tmp_path / "swing" / "annotations" / "alice"
        ann_dir.mkdir(parents=True)
        (ann_dir / "a.beats").write_text("1.0 1")
        assert list_migrated_track_ids("swing") == []


# ---------------------------------------------------------------------------
# resolve_track_audio
# ---------------------------------------------------------------------------


class TestResolveTrackAudio:
    def test_existing_file_returns_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path)
        tracks_dir = tmp_path / "swing" / "tracks"
        tracks_dir.mkdir(parents=True)
        (tracks_dir / "a.wav").write_bytes(b"")
        assert resolve_track_audio("swing", "a") == tracks_dir / "a.wav"

    def test_missing_file_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path)
        assert resolve_track_audio("swing", "missing") is None

    def test_missing_tracks_dir_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dataformats, "DATA_DIR", tmp_path)
        (tmp_path / "swing").mkdir()
        assert resolve_track_audio("swing", "a") is None


# ---------------------------------------------------------------------------
# bpm_stats
# ---------------------------------------------------------------------------


class TestBpmStats:
    def test_fewer_than_two_beats_returns_none_triple(self):
        assert bpm_stats(np.array([1.0])) == (None, None, None)
        assert bpm_stats(np.array([])) == (None, None, None)

    def test_constant_tempo(self):
        # 0.5s intervals -> 120 BPM exactly, every stat should agree.
        beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mean, median, std = bpm_stats(beat_times)
        assert mean == pytest.approx(120.0)
        assert median == pytest.approx(120.0)
        assert std == pytest.approx(0.0)

    def test_varying_tempo_mean_median_std(self):
        # intervals 0.5s, 1.0s -> tempos 120, 60 BPM.
        beat_times = np.array([0.0, 0.5, 1.5])
        mean, median, std = bpm_stats(beat_times)
        assert mean == pytest.approx(90.0)
        assert median == pytest.approx(90.0)
        assert std == pytest.approx(30.0)
