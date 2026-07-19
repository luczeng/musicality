"""Tests for tools.annotator.data's track metadata persistence."""

import tools.annotator.data as annotator_data
from tools.annotator.data import (
    TrackMetadata,
    load_metadata,
    metadata_path,
    save_metadata,
)


class TestMetadataPath:
    def test_path_matches_annotations_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        path = metadata_path("swing", "take1")
        assert path == tmp_path / "swing" / "annotations" / "take1.meta.json"


class TestSaveLoadMetadata:
    def test_round_trips_all_fields(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        metadata = TrackMetadata(
            location="The Blue Room", device="iPhone 13 mini", structure="blues"
        )
        save_metadata("swing", "take1", metadata)
        assert load_metadata("swing", "take1") == metadata

    def test_round_trips_partial_fields(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        metadata = TrackMetadata(structure="swing")
        save_metadata("swing", "take2", metadata)
        assert load_metadata("swing", "take2") == metadata

    def test_creates_parent_dirs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        save_metadata("new_dataset", "take1", TrackMetadata(device="desktop"))
        assert (tmp_path / "new_dataset" / "annotations" / "take1.meta.json").exists()

    def test_missing_metadata_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(annotator_data, "DATA_DIR", tmp_path)
        assert load_metadata("swing", "never_saved") is None
